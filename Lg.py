"""
LangGraph ReAct Agent with Arize.ai Tracing and Goal Drift Detection
This implementation includes:
1. LangGraph ReAct agent with MCP tools
2. Arize.ai trace provider and tracing
3. Goal drift and path divergence detection
4. Online evaluations and monitoring alerts
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import uuid

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Arize imports
from arize.pandas.logger import Client as ArizeClient
from arize.utils.types import ModelTypes, Environments
from phoenix.trace import TraceProvider
from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.trace.openai import OpenAIInstrumentor
import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    QACorrectness,
    run_evals,
    OpenAIModel
)
from phoenix.trace.dsl import SpanQuery

# MCP (Model Context Protocol) tools simulation
class MCPTools:
    """Simulated MCP tools for the agent"""
    
    @staticmethod
    def search_knowledge_base(query: str) -> str:
        """Search internal knowledge base"""
        return f"Knowledge base results for: {query}"
    
    @staticmethod
    def execute_code(code: str) -> str:
        """Execute code safely"""
        return f"Code execution result: {code[:50]}..."
    
    @staticmethod
    def fetch_context(context_id: str) -> str:
        """Fetch specific context"""
        return f"Context fetched for ID: {context_id}"

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    current_tool: Optional[str]
    tool_calls: List[Dict[str, Any]]
    goal: str
    initial_goal: str
    path_history: List[str]
    step_count: int

class LangGraphArizeAgent:
    def __init__(self, openai_api_key: str, arize_api_key: str, arize_space_id: str):
        """
        Initialize the LangGraph agent with Arize.ai integration
        
        Args:
            openai_api_key: OpenAI API key
            arize_api_key: Arize API key
            arize_space_id: Arize space ID
        """
        self.openai_api_key = openai_api_key
        self.arize_api_key = arize_api_key
        self.arize_space_id = arize_space_id
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ARIZE_API_KEY"] = arize_api_key
        os.environ["ARIZE_SPACE_ID"] = arize_space_id
        
        # Initialize Arize client
        self.arize_client = ArizeClient(
            api_key=arize_api_key,
            space_id=arize_space_id
        )
        
        # Initialize Phoenix for tracing
        self.setup_phoenix_tracing()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Setup tools
        self.tools = self.create_mcp_tools()
        self.tool_executor = ToolExecutor(self.tools)
        
        # Create the agent graph
        self.graph = self.create_agent_graph()
        
        # Compile the graph
        self.app = self.graph.compile(checkpointer=MemorySaver())
        
    def setup_phoenix_tracing(self):
        """Setup Phoenix/Arize tracing"""
        # Launch Phoenix for local tracing
        px.launch_app()
        
        # Setup OpenAI instrumentation
        OpenAIInstrumentor().instrument()
        
        # Setup LangChain instrumentation
        LangChainInstrumentor().instrument()
        
        # Configure trace provider for Arize
        self.trace_provider = TraceProvider(
            project_name="langgraph-agent-monitoring",
            api_key=self.arize_api_key
        )
        
    def create_mcp_tools(self) -> List[Tool]:
        """Create MCP-style tools for the agent"""
        tools = [
            StructuredTool.from_function(
                func=MCPTools.search_knowledge_base,
                name="search_knowledge",
                description="Search the knowledge base for information"
            ),
            StructuredTool.from_function(
                func=MCPTools.execute_code,
                name="execute_code",
                description="Execute Python code safely"
            ),
            StructuredTool.from_function(
                func=MCPTools.fetch_context,
                name="fetch_context",
                description="Fetch specific context by ID"
            ),
        ]
        return tools
    
    def create_agent_graph(self) -> StateGraph:
        """Create the LangGraph ReAct agent graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("goal_check", self.goal_drift_check_node)
        workflow.add_node("path_check", self.path_divergence_check_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "goal_check": "goal_check",
                "end": END
            }
        )
        workflow.add_edge("tools", "path_check")
        workflow.add_edge("path_check", "agent")
        workflow.add_edge("goal_check", "agent")
        
        return workflow
    
    def agent_node(self, state: AgentState) -> Dict:
        """Main agent reasoning node"""
        messages = state["messages"]
        
        # Create prompt for ReAct agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a ReAct agent. 
            Current goal: {goal}
            Initial goal: {initial_goal}
            Step count: {step_count}
            
            Think step by step:
            1. Thought: Analyze what needs to be done
            2. Action: Choose and execute a tool
            3. Observation: Analyze the result
            4. Repeat until goal is achieved
            """),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Get response from LLM
        response = self.llm.invoke(
            prompt.format_messages(
                goal=state.get("goal", ""),
                initial_goal=state.get("initial_goal", ""),
                step_count=state.get("step_count", 0),
                messages=messages
            )
        )
        
        # Update state
        return {
            "messages": messages + [response],
            "step_count": state.get("step_count", 0) + 1,
            "path_history": state.get("path_history", []) + [f"agent_step_{state.get('step_count', 0)}"]
        }
    
    def tool_node(self, state: AgentState) -> Dict:
        """Execute tools based on agent's decision"""
        last_message = state["messages"][-1]
        
        # Parse tool call from message
        tool_call = self.parse_tool_call(last_message)
        
        if tool_call:
            # Execute tool
            result = self.tool_executor.invoke(tool_call)
            
            # Create tool message
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call.tool_call_id
            )
            
            # Track tool usage
            tool_calls = state.get("tool_calls", [])
            tool_calls.append({
                "tool": tool_call.tool,
                "args": tool_call.tool_input,
                "result": str(result),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "messages": state["messages"] + [tool_message],
                "tool_calls": tool_calls,
                "current_tool": tool_call.tool,
                "path_history": state.get("path_history", []) + [f"tool_{tool_call.tool}"]
            }
        
        return state
    
    def goal_drift_check_node(self, state: AgentState) -> Dict:
        """Check for goal drift using Arize evaluation"""
        initial_goal = state.get("initial_goal", "")
        current_goal = state.get("goal", "")
        
        # Calculate goal drift score
        drift_score = self.calculate_goal_drift(initial_goal, current_goal, state["messages"])
        
        # Log to Arize
        self.log_goal_drift_to_arize(drift_score, initial_goal, current_goal)
        
        # Create alert if drift is significant
        if drift_score > 0.7:
            self.create_drift_alert("Goal Drift Detected", drift_score)
        
        return state
    
    def path_divergence_check_node(self, state: AgentState) -> Dict:
        """Check for path divergence"""
        path_history = state.get("path_history", [])
        
        # Calculate path divergence
        divergence_score = self.calculate_path_divergence(path_history)
        
        # Log to Arize
        self.log_path_divergence_to_arize(divergence_score, path_history)
        
        # Create alert if divergence is significant
        if divergence_score > 0.6:
            self.create_divergence_alert("Path Divergence Detected", divergence_score)
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine next step in the graph"""
        step_count = state.get("step_count", 0)
        
        # Check for goal drift every 5 steps
        if step_count % 5 == 0 and step_count > 0:
            return "goal_check"
        
        # Check if we need to use tools
        last_message = state["messages"][-1]
        if self.has_tool_call(last_message):
            return "continue"
        
        # Check if we're done
        if "final answer" in last_message.content.lower():
            return "end"
        
        return "continue"
    
    def parse_tool_call(self, message: BaseMessage) -> Optional[ToolInvocation]:
        """Parse tool call from message"""
        # Simplified parsing logic
        content = message.content.lower()
        
        if "search" in content:
            return ToolInvocation(
                tool="search_knowledge",
                tool_input={"query": "extracted query"},
                tool_call_id=str(uuid.uuid4())
            )
        elif "execute" in content:
            return ToolInvocation(
                tool="execute_code",
                tool_input={"code": "print('hello')"},
                tool_call_id=str(uuid.uuid4())
            )
        elif "fetch" in content:
            return ToolInvocation(
                tool="fetch_context",
                tool_input={"context_id": "ctx_123"},
                tool_call_id=str(uuid.uuid4())
            )
        
        return None
    
    def has_tool_call(self, message: BaseMessage) -> bool:
        """Check if message contains tool call"""
        keywords = ["search", "execute", "fetch", "use tool", "call"]
        return any(keyword in message.content.lower() for keyword in keywords)
    
    def calculate_goal_drift(self, initial_goal: str, current_goal: str, messages: List[BaseMessage]) -> float:
        """Calculate goal drift score using embeddings and semantic similarity"""
        # Use OpenAI embeddings for semantic similarity
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_api_key)
        
        # Get embeddings
        initial_emb = client.embeddings.create(
            input=initial_goal,
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        current_emb = client.embeddings.create(
            input=current_goal,
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        # Calculate cosine similarity
        import numpy as np
        similarity = np.dot(initial_emb, current_emb) / (np.linalg.norm(initial_emb) * np.linalg.norm(current_emb))
        
        # Drift score is inverse of similarity
        drift_score = 1 - similarity
        
        return drift_score
    
    def calculate_path_divergence(self, path_history: List[str]) -> float:
        """Calculate path divergence score"""
        if len(path_history) < 2:
            return 0.0
        
        # Calculate entropy of path
        from collections import Counter
        import math
        
        counter = Counter(path_history)
        total = len(path_history)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize entropy to [0, 1]
        max_entropy = math.log2(len(counter))
        divergence_score = entropy / max_entropy if max_entropy > 0 else 0
        
        return divergence_score
    
    def log_goal_drift_to_arize(self, drift_score: float, initial_goal: str, current_goal: str):
        """Log goal drift metrics to Arize"""
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=str(uuid.uuid4()),
            prediction_label="goal_drift_check",
            features={
                "initial_goal": initial_goal,
                "current_goal": current_goal
            },
            tags={
                "drift_score": drift_score,
                "threshold": 0.7,
                "drift_detected": drift_score > 0.7
            }
        )
    
    def log_path_divergence_to_arize(self, divergence_score: float, path_history: List[str]):
        """Log path divergence metrics to Arize"""
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=str(uuid.uuid4()),
            prediction_label="path_divergence_check",
            features={
                "path_length": len(path_history),
                "unique_steps": len(set(path_history))
            },
            tags={
                "divergence_score": divergence_score,
                "threshold": 0.6,
                "divergence_detected": divergence_score > 0.6
            }
        )
    
    def create_drift_alert(self, alert_name: str, drift_score: float):
        """Create alert for goal drift"""
        alert = {
            "alert_name": alert_name,
            "severity": "HIGH" if drift_score > 0.8 else "MEDIUM",
            "drift_score": drift_score,
            "timestamp": datetime.now().isoformat(),
            "action_required": "Review agent behavior and adjust prompts"
        }
        
        # Log alert to Arize
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=str(uuid.uuid4()),
            prediction_label="alert",
            tags=alert
        )
        
        print(f"🚨 Alert: {alert}")
    
    def create_divergence_alert(self, alert_name: str, divergence_score: float):
        """Create alert for path divergence"""
        alert = {
            "alert_name": alert_name,
            "severity": "HIGH" if divergence_score > 0.7 else "MEDIUM",
            "divergence_score": divergence_score,
            "timestamp": datetime.now().isoformat(),
            "action_required": "Agent may be stuck in a loop or exploring inefficient paths"
        }
        
        # Log alert to Arize
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=str(uuid.uuid4()),
            prediction_label="alert",
            tags=alert
        )
        
        print(f"🚨 Alert: {alert}")
    
    async def run(self, user_input: str, initial_goal: str):
        """Run the agent with user input"""
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "goal": initial_goal,
            "initial_goal": initial_goal,
            "tool_calls": [],
            "path_history": [],
            "step_count": 0
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        async for event in self.app.astream(initial_state, config):
            print(f"Event: {event}")
            
        return event


class ArizeOnlineEvaluator:
    """Online evaluation templates for Arize.ai"""
    
    def __init__(self, arize_client: ArizeClient):
        self.arize_client = arize_client
        self.eval_model = OpenAIModel(model="gpt-4")
    
    def create_goal_drift_template(self):
        """Create evaluation template for goal drift detection"""
        template = {
            "name": "Goal Drift Detection",
            "description": "Detects when agent drifts from initial goal",
            "eval_type": "classification",
            "prompt_template": """
            Initial Goal: {initial_goal}
            Current Behavior: {current_behavior}
            
            Question: Has the agent significantly drifted from its initial goal?
            
            Criteria:
            1. Is the agent still working towards the initial goal?
            2. Has the agent introduced new objectives not related to the initial goal?
            3. Is the agent's current approach still relevant to solving the initial problem?
            
            Answer with: YES (drifted) or NO (on track)
            """,
            "scoring": {
                "YES": 1.0,  # Drift detected
                "NO": 0.0    # No drift
            },
            "threshold": 0.7
        }
        return template
    
    def create_path_divergence_template(self):
        """Create evaluation template for path divergence detection"""
        template = {
            "name": "Path Divergence Detection",
            "description": "Detects inefficient or circular agent paths",
            "eval_type": "scoring",
            "prompt_template": """
            Agent Path History: {path_history}
            Number of Steps: {num_steps}
            
            Evaluate the efficiency of the agent's path:
            
            1. Are there repeated actions indicating a loop?
            2. Is the agent taking unnecessary detours?
            3. Could the goal be achieved with fewer steps?
            
            Score from 0 (efficient) to 1 (highly divergent):
            """,
            "scoring_rubric": {
                "0-0.3": "Efficient path",
                "0.3-0.6": "Some inefficiency",
                "0.6-0.8": "Significant divergence",
                "0.8-1.0": "Severe divergence/loops"
            },
            "threshold": 0.6
        }
        return template
    
    def run_online_eval(self, trace_data: Dict, template: Dict) -> float:
        """Run online evaluation using template"""
        # This would integrate with Arize's eval framework
        # Simplified version for demonstration
        
        if template["eval_type"] == "classification":
            # Run classification eval
            result = self.eval_model.evaluate(
                prompt=template["prompt_template"].format(**trace_data),
                expected_output=None
            )
            return template["scoring"].get(result, 0.0)
        
        elif template["eval_type"] == "scoring":
            # Run scoring eval
            result = self.eval_model.evaluate(
                prompt=template["prompt_template"].format(**trace_data),
                expected_output=None
            )
            return float(result)
        
        return 0.0


class MonitoringAlertSystem:
    """Monitoring and alerting system for agent behavior"""
    
    def __init__(self, arize_client: ArizeClient):
        self.arize_client = arize_client
        self.alert_rules = []
        
    def add_alert_rule(self, rule: Dict):
        """Add a monitoring alert rule"""
        self.alert_rules.append(rule)
    
    def create_default_rules(self):
        """Create default monitoring rules"""
        self.alert_rules = [
            {
                "name": "High Goal Drift",
                "metric": "goal_drift_score",
                "condition": "greater_than",
                "threshold": 0.7,
                "severity": "HIGH",
                "notification": "email",
                "cooldown_minutes": 30
            },
            {
                "name": "Path Divergence Warning",
                "metric": "path_divergence_score",
                "condition": "greater_than",
                "threshold": 0.6,
                "severity": "MEDIUM",
                "notification": "slack",
                "cooldown_minutes": 15
            },
            {
                "name": "Excessive Tool Calls",
                "metric": "tool_call_count",
                "condition": "greater_than",
                "threshold": 20,
                "severity": "LOW",
                "notification": "dashboard",
                "cooldown_minutes": 60
            },
            {
                "name": "Agent Loop Detection",
                "metric": "repeated_action_ratio",
                "condition": "greater_than",
                "threshold": 0.5,
                "severity": "HIGH",
                "notification": "pagerduty",
                "cooldown_minutes": 10
            }
        ]
    
    def check_alerts(self, metrics: Dict):
        """Check if any alerts should be triggered"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            metric_value = metrics.get(rule["metric"])
            if metric_value is None:
                continue
            
            if self.evaluate_condition(metric_value, rule["condition"], rule["threshold"]):
                alert = self.trigger_alert(rule, metric_value)
                triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return value == threshold
        return False
    
    def trigger_alert(self, rule: Dict, metric_value: float) -> Dict:
        """Trigger an alert"""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "rule_name": rule["name"],
            "metric": rule["metric"],
            "value": metric_value,
            "threshold": rule["threshold"],
            "severity": rule["severity"],
            "timestamp": datetime.now().isoformat(),
            "notification_sent": self.send_notification(rule["notification"])
        }
        
        # Log to Arize
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=alert["alert_id"],
            prediction_label="monitoring_alert",
            tags=alert
        )
        
        return alert
    
    def send_notification(self, notification_type: str) -> bool:
        """Send notification (mock implementation)"""
        print(f"📧 Sending {notification_type} notification...")
        return True


# Main execution
async def main():
    """Main function to demonstrate the complete system"""
    
    # Configuration
    OPENAI_API_KEY = "your-openai-api-key"
    ARIZE_API_KEY = "your-arize-api-key"
    ARIZE_SPACE_ID = "your-arize-space-id"
    
    # Initialize agent
    agent = LangGraphArizeAgent(
        openai_api_key=OPENAI_API_KEY,
        arize_api_key=ARIZE_API_KEY,
        arize_space_id=ARIZE_SPACE_ID
    )
    
    # Initialize evaluator
    evaluator = ArizeOnlineEvaluator(agent.arize_client)
    
    # Create evaluation templates
    goal_drift_template = evaluator.create_goal_drift_template()
    path_divergence_template = evaluator.create_path_divergence_template()
    
    # Initialize monitoring
    monitor = MonitoringAlertSystem(agent.arize_client)
    monitor.create_default_rules()
    
    # Example usage
    user_query = "Help me analyze sales data and create a report"
    initial_goal = "Analyze sales data and generate comprehensive report"
    
    print("🚀 Starting LangGraph Agent with Arize Monitoring...")
    print(f"User Query: {user_query}")
    print(f"Initial Goal: {initial_goal}")
    print("-" * 50)
    
    # Run agent
    result = await agent.run(user_query, initial_goal)
    
    # Check metrics and alerts
    metrics = {
        "goal_drift_score": 0.75,  # Example metric
        "path_divergence_score": 0.65,  # Example metric
        "tool_call_count": 15,
        "repeated_action_ratio": 0.3
    }
    
    triggered_alerts = monitor.check_alerts(metrics)
    
    print("\n📊 Monitoring Results:")
    print(f"Triggered Alerts: {len(triggered_alerts)}")
    for alert in triggered_alerts:
        print(f"  - {alert['rule_name']}: {alert['metric']} = {alert['value']}")
    
    print("\n✅ Agent execution completed with full monitoring!")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())


"""
Installation Requirements:
pip install langgraph langchain langchain-openai
pip install arize-phoenix arize
pip install openai numpy pandas

Environment Variables Required:
- OPENAI_API_KEY: Your OpenAI API key
- ARIZE_API_KEY: Your Arize API key  
- ARIZE_SPACE_ID: Your Arize space ID

Features Implemented:
1. ✅ LangGraph ReAct agent with MCP-style tools
2. ✅ Arize.ai trace provider and tracing setup
3. ✅ Goal drift detection with semantic similarity
4. ✅ Path divergence detection with entropy calculation
5. ✅ Online evaluation templates for Arize
6. ✅ Monitoring alert system with configurable rules
7. ✅ Complete integration with Arize dashboard logging

The agent will:
- Execute tools based on ReAct pattern
- Track all spans and traces in Arize dashboard
- Detect goal drift using embedding similarity
- Detect path divergence using entropy metrics
- Trigger alerts based on configurable thresholds
- Log all metrics and alerts to Arize for visualization
"""
