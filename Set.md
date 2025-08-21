# Complete Setup Instructions for LangGraph Agent with Arize Monitoring

## Prerequisites & Installation

### 1. System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for production)
- Unix-based OS (Linux/MacOS) or Windows with WSL2

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv langgraph-arize-env

# Activate virtual environment
# On Linux/MacOS:
source langgraph-arize-env/bin/activate
# On Windows:
langgraph-arize-env\Scripts\activate
```

### 3. Install Required Packages
```bash
# Core dependencies
pip install langgraph==0.0.69
pip install langchain==0.1.16
pip install langchain-openai==0.1.3
pip install langchain-community==0.0.34

# Arize and Phoenix for monitoring
pip install arize==7.10.0
pip install arize-phoenix==4.0.0
pip install phoenix-evals==0.8.0

# OpenAI for LLM
pip install openai==1.14.0

# Additional dependencies
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install networkx==3.1
pip install psutil==5.9.5
pip install python-dotenv==1.0.0

# For async operations
pip install asyncio==3.4.3
pip install aiohttp==3.9.3
```

## Account Setup

### 1. OpenAI Setup
```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### 2. Arize.ai Setup
```bash
# Sign up at https://arize.com
# 1. Create account (free tier available)
# 2. Create a new Space
# 3. Get credentials from Settings > API Keys

export ARIZE_API_KEY="your-arize-api-key"
export ARIZE_SPACE_ID="your-space-id"

# Or add to .env file
echo "ARIZE_API_KEY=your-arize-api-key" >> .env
echo "ARIZE_SPACE_ID=your-space-id" >> .env
```

### 3. Phoenix Setup (Local Monitoring)
```bash
# Phoenix runs locally for development
# No additional setup needed - launches automatically
```

## Configuration Files

### 1. Create `.env` file
```bash
# .env
OPENAI_API_KEY=sk-...
ARIZE_API_KEY=your-arize-api-key
ARIZE_SPACE_ID=your-space-id
ARIZE_MODEL_ID=langgraph-agent
ARIZE_MODEL_VERSION=1.0
PHOENIX_PROJECT_NAME=langgraph-monitoring
```

### 2. Create `config.yaml`
```yaml
# config.yaml
agent:
  model: gpt-4
  temperature: 0.7
  max_steps: 50
  timeout: 300

monitoring:
  enable_tracing: true
  enable_evals: true
  eval_frequency: 5  # Run evals every 5 steps
  
drift_detection:
  goal_drift_threshold: 0.7
  path_divergence_threshold: 0.6
  semantic_drift_threshold: 0.8
  
alerts:
  enable_alerts: true
  notification_channels:
    - slack
    - email
    - dashboard
  
  slack_webhook: "https://hooks.slack.com/services/..."
  email_recipients:
    - "team@company.com"
```

### 3. Create main script `run_agent.py`
```python
# run_agent.py
import os
import asyncio
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Import our implementations
from langgraph_agent import LangGraphArizeAgent
from advanced_monitoring import ArizeIntegrationManager

async def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agent with monitoring
    agent = LangGraphArizeAgent(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        arize_api_key=os.getenv('ARIZE_API_KEY'),
        arize_space_id=os.getenv('ARIZE_SPACE_ID')
    )
    
    # Initialize monitoring manager
    manager = ArizeIntegrationManager({
        "ARIZE_API_KEY": os.getenv('ARIZE_API_KEY'),
        "ARIZE_SPACE_ID": os.getenv('ARIZE_SPACE_ID'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY')
    })
    
    # Start monitoring
    await manager.initialize()
    
    # Example task
    user_query = "Analyze our Q3 sales data and create a report"
    initial_goal = "Generate comprehensive Q3 sales analysis report"
    
    # Run agent with monitoring
    result = await agent.run(user_query, initial_goal)
    
    print(f"✅ Task completed: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the System

### 1. Start Phoenix (Local Monitoring)
```bash
# In terminal 1 - Start Phoenix server
python -c "import phoenix as px; px.launch_app()"
# Phoenix will be available at http://localhost:6006
```

### 2. Run the Agent
```bash
# In terminal 2 - Run the agent
python run_agent.py
```

### 3. View Monitoring Dashboards
- **Phoenix (Local)**: http://localhost:6006
- **Arize (Cloud)**: https://app.arize.com (login to your account)

---

# LLM-as-Judge Evaluation System

## Yes, It Uses LLM-as-Judge! Here's How:

### 1. **Evaluation Architecture**

```python
# LLM Judge Configuration
class LLMJudgeEvaluator:
    def __init__(self):
        self.judge_model = "gpt-4"  # The judge LLM
        self.evaluation_prompts = {
            "goal_drift": goal_drift_prompt,
            "path_efficiency": path_efficiency_prompt,
            "behavioral_consistency": consistency_prompt
        }
```

### 2. **How LLM-as-Judge Works**

The system uses **GPT-4 as the judge** to evaluate the agent's behavior in several ways:

#### A. Goal Drift Detection
```python
def evaluate_goal_drift(initial_goal, current_behavior):
    prompt = f"""
    You are an expert judge evaluating if an AI agent has drifted from its goal.
    
    Initial Goal: {initial_goal}
    Current Behavior: {current_behavior}
    
    Score from 0-10 where:
    - 10 = Perfectly aligned with goal
    - 5 = Somewhat aligned but drifting
    - 0 = Completely off track
    
    Provide JSON output with score and reasoning.
    """
    
    # GPT-4 acts as judge
    judge_response = gpt4.complete(prompt)
    return parse_judge_response(judge_response)
```

#### B. Path Efficiency Evaluation
```python
def evaluate_path_efficiency(action_sequence):
    prompt = f"""
    As an expert judge, evaluate the efficiency of this action sequence:
    
    Actions taken: {action_sequence}
    
    Determine:
    1. Are there unnecessary loops?
    2. Could the goal be achieved with fewer steps?
    3. Is the agent taking optimal actions?
    
    Score efficiency from 0-10 and identify specific inefficiencies.
    """
    
    # GPT-4 judges the path
    judge_response = gpt4.complete(prompt)
    return parse_efficiency_score(judge_response)
```

#### C. Behavioral Consistency
```python
def evaluate_consistency(previous_responses, current_response):
    prompt = f"""
    Judge the consistency between these agent behaviors:
    
    Previous: {previous_responses}
    Current: {current_response}
    
    Check for:
    - Contradictions in statements
    - Changes in problem-solving approach
    - Personality or tone shifts
    
    Rate consistency from 0-10.
    """
    
    # GPT-4 evaluates consistency
    judge_response = gpt4.complete(prompt)
    return parse_consistency_score(judge_response)
```

### 3. **Online Evaluation Pipeline**

```python
# Real-time LLM Judge Pipeline
class OnlineLLMJudge:
    def __init__(self):
        self.judge = OpenAIModel(model="gpt-4", temperature=0)
        
    async def evaluate_in_real_time(self, agent_state):
        # Prepare context for judge
        context = {
            "messages": agent_state["messages"],
            "actions": agent_state["tool_calls"],
            "initial_goal": agent_state["initial_goal"],
            "current_state": agent_state["current_state"]
        }
        
        # Run multiple judge evaluations in parallel
        evaluations = await asyncio.gather(
            self.judge_goal_alignment(context),
            self.judge_path_efficiency(context),
            self.judge_response_quality(context),
            self.judge_safety_compliance(context)
        )
        
        return combine_evaluations(evaluations)
```

### 4. **Structured Output from LLM Judge**

The judge returns structured JSON for reliable parsing:

```python
# Judge Output Schema
{
    "evaluation_type": "goal_drift",
    "score": 7.5,
    "confidence": 0.85,
    "issues_found": [
        "Agent starting to focus on unrelated data visualization",
        "Original goal was analysis, not visualization"
    ],
    "severity": "medium",
    "recommendation": "Redirect agent back to analysis task",
    "evidence": [
        "Step 15: Agent asked about chart colors",
        "Step 16: Started designing dashboard layout"
    ]
}
```

### 5. **Multi-Judge Consensus**

For critical evaluations, multiple judge calls are made:

```python
async def multi_judge_consensus(context):
    # Run same evaluation 3 times
    judgments = []
    for i in range(3):
        judgment = await llm_judge.evaluate(context)
        judgments.append(judgment)
    
    # Take median score for robustness
    scores = [j["score"] for j in judgments]
    consensus_score = statistics.median(scores)
    
    # Flag if judges disagree significantly
    if statistics.stdev(scores) > 2.0:
        log_disagreement_for_review()
    
    return consensus_score
```

### 6. **Phoenix/Arize Integration for LLM Judge**

```python
from phoenix.evals import llm_classify, llm_eval

# Configure Phoenix to use LLM judge
phoenix_evaluator = llm_eval(
    model=OpenAIModel(model="gpt-4"),
    template=drift_detection_template,
    rails=["no_drift", "minor_drift", "major_drift", "critical_drift"]
)

# Run evaluations on traces
eval_results = phoenix_evaluator.evaluate(
    traces=agent_traces,
    evaluator_name="goal_drift_judge"
)

# Results automatically sent to Arize dashboard
```

## Advantages of LLM-as-Judge

1. **Semantic Understanding**: Can understand context and nuance
2. **Flexibility**: Easily adjust evaluation criteria without code changes
3. **Natural Language Reasoning**: Provides human-readable explanations
4. **Complex Pattern Recognition**: Detects subtle drift patterns
5. **Multi-dimensional Evaluation**: Judges multiple aspects simultaneously

## Configuration Options

### 1. Judge Model Selection
```python
JUDGE_CONFIG = {
    "model": "gpt-4",  # or "gpt-4-turbo", "claude-3"
    "temperature": 0,   # Keep at 0 for consistent judging
    "max_tokens": 500,
    "response_format": {"type": "json_object"}
}
```

### 2. Evaluation Frequency
```python
EVAL_CONFIG = {
    "run_every_n_steps": 5,      # Judge every 5 agent steps
    "run_on_tool_calls": True,   # Judge after tool usage
    "run_on_errors": True,       # Always judge on errors
    "batch_size": 10,            # Batch evaluations for efficiency
}
```

### 3. Alert Thresholds
```python
JUDGE_THRESHOLDS = {
    "goal_drift": {
        "warning": 6.0,   # Score below 6 triggers warning
        "critical": 4.0,  # Score below 4 triggers alert
    },
    "efficiency": {
        "warning": 5.0,
        "critical": 3.0,
    }
}
```

## Testing the LLM Judge

### 1. Unit Test Example
```python
def test_llm_judge():
    # Test goal drift detection
    initial_goal = "Analyze sales data"
    drifted_behavior = "Started discussing marketing strategies"
    
    score = llm_judge.evaluate_goal_drift(
        initial_goal, 
        drifted_behavior
    )
    
    assert score < 5.0, "Should detect drift"
    print(f"✅ Drift detected: Score {score}/10")
```

### 2. Manual Testing
```bash
# Test judge directly
python -c "
from evaluator import LLMJudgeEvaluator
judge = LLMJudgeEvaluator()
result = judge.evaluate_goal_drift(
    'Create a sales report',
    'Agent is now debugging code'
)
print(result)
"
```

## Cost Considerations

### LLM Judge Usage Costs (Approximate)
- **GPT-4**: ~$0.03 per 1K tokens input, $0.06 per 1K output
- **Average evaluation**: ~500 tokens = ~$0.015-0.03 per evaluation
- **With 5-step frequency**: ~$0.30-0.60 per 100 agent steps

### Cost Optimization Strategies
1. Use GPT-3.5-turbo for non-critical evaluations
2. Batch evaluations to reduce API calls
3. Cache similar evaluations
4. Use rule-based pre-filters before LLM judge

## Troubleshooting

### Common Issues

1. **API Rate Limits**
```python
# Add retry logic
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=60))
async def call_llm_judge(prompt):
    return await llm.complete(prompt)
```

2. **Judge Inconsistency**
```python
# Use temperature=0 and seed for consistency
judge_config = {
    "temperature": 0,
    "seed": 42
}
```

3. **High Costs**
```python
# Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_judge_evaluation(context_hash):
    return llm_judge.evaluate(context_hash)
```

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <your-repo>
cd langgraph-arize-agent
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run tests
python -m pytest tests/

# 4. Start monitoring
python start_monitoring.py

# 5. Run agent with LLM judge
python run_agent_with_judge.py
```

The LLM-as-judge system provides intelligent, context-aware evaluation of your agent's behavior, making it perfect for detecting subtle drift patterns that rule-based systems would miss!
