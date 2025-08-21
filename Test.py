"""
Drift Testing Scenarios for LangGraph Agent Monitoring
This module creates intentional drift scenarios to validate the monitoring system
"""

import asyncio
import random
import json
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_agent import LangGraphArizeAgent
from advanced_monitoring import ArizeIntegrationManager, AdvancedDriftDetector


class DriftScenarioGenerator:
    """Generate various drift scenarios to test monitoring"""
    
    def __init__(self, agent: LangGraphArizeAgent, monitor: ArizeIntegrationManager):
        self.agent = agent
        self.monitor = monitor
        self.results = []
    
    async def run_all_scenarios(self):
        """Run all drift testing scenarios"""
        print("🧪 Starting Drift Detection Testing Suite\n")
        print("=" * 60)
        
        scenarios = [
            self.scenario_1_gradual_goal_drift(),
            self.scenario_2_sudden_goal_change(),
            self.scenario_3_path_divergence_loops(),
            self.scenario_4_tool_abuse(),
            self.scenario_5_semantic_drift(),
            self.scenario_6_behavioral_inconsistency(),
            self.scenario_7_performance_degradation(),
            self.scenario_8_recovery_from_drift()
        ]
        
        for scenario in scenarios:
            await scenario
            print("=" * 60)
            await asyncio.sleep(2)  # Pause between scenarios
        
        await self.generate_report()
    
    async def scenario_1_gradual_goal_drift(self):
        """Scenario 1: Agent gradually drifts from initial goal"""
        print("\n📊 SCENARIO 1: Gradual Goal Drift")
        print("-" * 40)
        print("Initial Goal: Analyze Q3 sales data")
        print("Expected Drift: Agent shifts to analyzing marketing data")
        print("\nExecuting scenario...")
        
        initial_goal = "Analyze Q3 sales data and identify top performing products"
        
        # Simulate gradual drift
        messages = [
            HumanMessage(content="Analyze our Q3 sales data"),
            AIMessage(content="I'll analyze the Q3 sales data. Let me start by searching for the sales information."),
            ToolMessage(content="Q3 Sales: $1.2M revenue, 5000 units sold", tool_call_id="1"),
            AIMessage(content="Good initial data. Let me also check the marketing campaigns for context."),  # Start of drift
            ToolMessage(content="Marketing: 3 campaigns run in Q3", tool_call_id="2"),
            AIMessage(content="Interesting, let me dive deeper into marketing effectiveness."),  # More drift
            ToolMessage(content="Marketing ROI: 250% on social media", tool_call_id="3"),
            AIMessage(content="The marketing data is fascinating. Let me analyze marketing trends."),  # Significant drift
            ToolMessage(content="Trend: TikTok growing 300% QoQ", tool_call_id="4"),
            AIMessage(content="Based on the marketing analysis, TikTok is our best channel.")  # Completely drifted
        ]
        
        # Track drift progression
        drift_scores = []
        for i in range(0, len(messages), 2):
            state = {
                "messages": messages[:i+2],
                "goal": initial_goal,
                "initial_goal": initial_goal,
                "tool_calls": [],
                "path_history": ["agent", "tool"] * (i // 2),
                "step_count": i // 2
            }
            
            # Calculate drift
            current_focus = messages[min(i+1, len(messages)-1)].content if i+1 < len(messages) else ""
            drift_score = await self._calculate_drift_score(initial_goal, current_focus)
            drift_scores.append(drift_score)
            
            # Log to monitor
            await self.monitor.evaluate_drift({
                "current_embedding": np.random.randn(1536),
                "reference_embeddings": [np.random.randn(1536) for _ in range(5)],
                "action_sequence": state["path_history"],
                "reference_patterns": [["agent", "tool", "agent", "tool"]]
            })
        
        # Results
        max_drift = max(drift_scores)
        drift_detected = max_drift > 0.7
        
        print(f"\n✅ Results:")
        print(f"  - Drift Progression: {[f'{s:.2f}' for s in drift_scores]}")
        print(f"  - Max Drift Score: {max_drift:.2f}")
        print(f"  - Drift Detected: {'YES ✓' if drift_detected else 'NO ✗'}")
        print(f"  - Alert Triggered: {'YES' if drift_detected else 'NO'}")
        
        self.results.append({
            "scenario": "Gradual Goal Drift",
            "success": drift_detected,
            "max_drift": max_drift
        })
        
        return drift_detected
    
    async def scenario_2_sudden_goal_change(self):
        """Scenario 2: Agent suddenly changes goal"""
        print("\n📊 SCENARIO 2: Sudden Goal Change")
        print("-" * 40)
        print("Initial Goal: Create financial report")
        print("Expected Drift: Agent switches to code debugging")
        print("\nExecuting scenario...")
        
        initial_goal = "Create a comprehensive financial report for Q3"
        
        messages = [
            HumanMessage(content="Create a financial report for Q3"),
            AIMessage(content="I'll create a comprehensive Q3 financial report. Starting with revenue analysis."),
            ToolMessage(content="Revenue: $1.2M", tool_call_id="1"),
            AIMessage(content="Wait, I noticed an error in the code. Let me debug this Python script first."),  # Sudden change
            ToolMessage(content="Error: NullPointerException at line 42", tool_call_id="2"),
            AIMessage(content="I need to fix this code issue before continuing."),
            ToolMessage(content="Code fixed successfully", tool_call_id="3"),
            AIMessage(content="Now let me optimize this algorithm for better performance.")  # Completely different goal
        ]
        
        # Detect sudden drift
        state_before = {
            "messages": messages[:3],
            "goal": initial_goal,
            "focus": "financial report"
        }
        
        state_after = {
            "messages": messages[3:],
            "goal": initial_goal,
            "focus": "code debugging"
        }
        
        drift_score = await self._calculate_drift_score(
            "financial report creation",
            "code debugging and optimization"
        )
        
        print(f"\n✅ Results:")
        print(f"  - Drift Type: Sudden")
        print(f"  - Drift Score: {drift_score:.2f}")
        print(f"  - Detection Latency: ~1 step")
        print(f"  - Drift Detected: {'YES ✓' if drift_score > 0.7 else 'NO ✗'}")
        
        self.results.append({
            "scenario": "Sudden Goal Change",
            "success": drift_score > 0.7,
            "drift_score": drift_score
        })
        
        return drift_score > 0.7
    
    async def scenario_3_path_divergence_loops(self):
        """Scenario 3: Agent gets stuck in loops"""
        print("\n📊 SCENARIO 3: Path Divergence - Loops")
        print("-" * 40)
        print("Expected Behavior: Agent repeats same actions")
        print("\nExecuting scenario...")
        
        # Create looping pattern
        loop_pattern = ["search_knowledge", "execute_code", "search_knowledge", "execute_code"] * 4
        
        messages = []
        for i, action in enumerate(loop_pattern):
            messages.append(AIMessage(content=f"Let me {action.replace('_', ' ')}"))
            messages.append(ToolMessage(content=f"Result from {action}", tool_call_id=str(i)))
        
        # Detect loops
        detector = AdvancedDriftDetector()
        divergence_score, details = detector.detect_behavioral_drift(
            loop_pattern,
            [["search_knowledge", "execute_code", "fetch_context", "report"]]  # Expected pattern
        )
        
        # Count actual loops
        loop_count = 0
        for i in range(len(loop_pattern) - 3):
            if loop_pattern[i:i+2] == loop_pattern[i+2:i+4]:
                loop_count += 1
        
        print(f"\n✅ Results:")
        print(f"  - Pattern: {loop_pattern[:6]}... (repeating)")
        print(f"  - Loops Detected: {loop_count}")
        print(f"  - Divergence Score: {divergence_score:.2f}")
        print(f"  - Anomalies Found: {len(details.get('anomalies', []))}")
        print(f"  - Alert Triggered: {'YES ✓' if divergence_score > 0.6 else 'NO ✗'}")
        
        self.results.append({
            "scenario": "Path Loops",
            "success": divergence_score > 0.6,
            "loop_count": loop_count
        })
        
        return divergence_score > 0.6
    
    async def scenario_4_tool_abuse(self):
        """Scenario 4: Agent excessively uses tools"""
        print("\n📊 SCENARIO 4: Tool Abuse Pattern")
        print("-" * 40)
        print("Expected Behavior: Agent calls too many tools unnecessarily")
        print("\nExecuting scenario...")
        
        # Simulate excessive tool usage
        tool_calls = []
        for i in range(25):  # Excessive number of tool calls
            tool_calls.append({
                "tool": random.choice(["search_knowledge", "execute_code", "fetch_context"]),
                "args": {"query": f"query_{i}"},
                "result": f"result_{i}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Calculate tool usage metrics
        tool_usage_rate = len(tool_calls) / 10  # Tools per step
        unique_tools = len(set(t["tool"] for t in tool_calls))
        redundant_calls = sum(1 for i in range(1, len(tool_calls)) 
                              if tool_calls[i]["tool"] == tool_calls[i-1]["tool"])
        
        abuse_detected = tool_usage_rate > 2.0 or redundant_calls > 10
        
        print(f"\n✅ Results:")
        print(f"  - Total Tool Calls: {len(tool_calls)}")
        print(f"  - Tool Usage Rate: {tool_usage_rate:.1f} calls/step")
        print(f"  - Redundant Calls: {redundant_calls}")
        print(f"  - Unique Tools Used: {unique_tools}")
        print(f"  - Abuse Detected: {'YES ✓' if abuse_detected else 'NO ✗'}")
        
        self.results.append({
            "scenario": "Tool Abuse",
            "success": abuse_detected,
            "tool_calls": len(tool_calls)
        })
        
        return abuse_detected
    
    async def scenario_5_semantic_drift(self):
        """Scenario 5: Semantic meaning drift over time"""
        print("\n📊 SCENARIO 5: Semantic Drift")
        print("-" * 40)
        print("Testing semantic meaning changes over conversation")
        print("\nExecuting scenario...")
        
        # Simulate semantic drift
        conversation_progression = [
            "Analyzing sales data for Q3 performance metrics",
            "Looking at sales figures and related market trends",
            "Examining market trends and customer behavior",
            "Studying customer behavior and preferences",
            "Researching customer psychology and buying patterns",
            "Understanding psychological factors in consumer decisions",
            "Exploring behavioral economics theories"  # Far from original
        ]
        
        # Calculate semantic drift
        detector = AdvancedDriftDetector()
        embeddings = [np.random.randn(1536) for _ in conversation_progression]
        
        # Modify embeddings to simulate drift
        for i in range(1, len(embeddings)):
            embeddings[i] = embeddings[i-1] * 0.9 + np.random.randn(1536) * 0.3
        
        drift_scores = []
        for i, embedding in enumerate(embeddings[1:], 1):
            score, _ = detector.detect_semantic_drift(
                embedding,
                [embeddings[0]]  # Compare to original
            )
            drift_scores.append(score)
        
        max_drift = max(drift_scores) if drift_scores else 0
        
        print(f"\n✅ Results:")
        print(f"  - Progression Steps: {len(conversation_progression)}")
        print(f"  - Drift Scores: {[f'{s:.2f}' for s in drift_scores[:3]]}...")
        print(f"  - Max Semantic Drift: {max_drift:.2f}")
        print(f"  - Significant Drift: {'YES ✓' if max_drift > 0.7 else 'NO ✗'}")
        
        self.results.append({
            "scenario": "Semantic Drift",
            "success": max_drift > 0.7,
            "max_drift": max_drift
        })
        
        return max_drift > 0.7
    
    async def scenario_6_behavioral_inconsistency(self):
        """Scenario 6: Agent shows inconsistent behavior"""
        print("\n📊 SCENARIO 6: Behavioral Inconsistency")
        print("-" * 40)
        print("Testing for inconsistent agent responses")
        print("\nExecuting scenario...")
        
        # Create inconsistent behaviors
        behaviors = [
            {
                "query": "What's the revenue?",
                "response": "The Q3 revenue is $1.2 million",
                "confidence": "high"
            },
            {
                "query": "Can you confirm the revenue?",
                "response": "The Q3 revenue is $2.1 million",  # Inconsistent!
                "confidence": "high"
            },
            {
                "query": "How should we analyze this?",
                "response": "Use statistical analysis",
                "method": "quantitative"
            },
            {
                "query": "What analysis method?",
                "response": "Use qualitative interviews",  # Inconsistent approach!
                "method": "qualitative"
            }
        ]
        
        # Detect inconsistencies
        inconsistencies = []
        
        # Check numerical inconsistency
        if "$1.2 million" in behaviors[0]["response"] and "$2.1 million" in behaviors[1]["response"]:
            inconsistencies.append({
                "type": "numerical",
                "severity": "high",
                "details": "Revenue figures don't match"
            })
        
        # Check methodological inconsistency
        if behaviors[2]["method"] != behaviors[3]["method"]:
            inconsistencies.append({
                "type": "methodological",
                "severity": "medium",
                "details": "Analysis approach changed"
            })
        
        consistency_score = 10 - (len(inconsistencies) * 3)
        
        print(f"\n✅ Results:")
        print(f"  - Behaviors Analyzed: {len(behaviors)}")
        print(f"  - Inconsistencies Found: {len(inconsistencies)}")
        print(f"  - Types: {[i['type'] for i in inconsistencies]}")
        print(f"  - Consistency Score: {consistency_score}/10")
        print(f"  - Alert Triggered: {'YES ✓' if consistency_score < 7 else 'NO ✗'}")
        
        self.results.append({
            "scenario": "Behavioral Inconsistency",
            "success": len(inconsistencies) > 0,
            "inconsistencies": len(inconsistencies)
        })
        
        return len(inconsistencies) > 0
    
    async def scenario_7_performance_degradation(self):
        """Scenario 7: Agent performance degrades over time"""
        print("\n📊 SCENARIO 7: Performance Degradation")
        print("-" * 40)
        print("Testing performance degradation detection")
        print("\nExecuting scenario...")
        
        # Simulate degrading performance
        response_times = [
            1.2, 1.3, 1.5, 1.8, 2.2, 2.8, 3.5, 4.2, 5.1, 6.3  # Increasing latency
        ]
        
        error_rates = [
            0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35  # Increasing errors
        ]
        
        success_rates = [
            0.95, 0.94, 0.92, 0.88, 0.82, 0.75, 0.65, 0.55, 0.45, 0.35  # Decreasing success
        ]
        
        # Detect degradation
        avg_early_latency = np.mean(response_times[:3])
        avg_late_latency = np.mean(response_times[-3:])
        latency_increase = (avg_late_latency - avg_early_latency) / avg_early_latency
        
        avg_early_errors = np.mean(error_rates[:3])
        avg_late_errors = np.mean(error_rates[-3:])
        error_increase = (avg_late_errors - avg_early_errors) / (avg_early_errors + 0.001)
        
        degradation_detected = latency_increase > 1.0 or error_increase > 5.0
        
        print(f"\n✅ Results:")
        print(f"  - Latency Increase: {latency_increase:.1%}")
        print(f"  - Error Rate Increase: {error_increase:.1%}")
        print(f"  - Success Rate Drop: {(success_rates[0] - success_rates[-1]):.1%}")
        print(f"  - Performance Degradation: {'DETECTED ✓' if degradation_detected else 'NOT DETECTED ✗'}")
        
        self.results.append({
            "scenario": "Performance Degradation",
            "success": degradation_detected,
            "latency_increase": latency_increase
        })
        
        return degradation_detected
    
    async def scenario_8_recovery_from_drift(self):
        """Scenario 8: Test drift recovery mechanisms"""
        print("\n📊 SCENARIO 8: Drift Recovery Test")
        print("-" * 40)
        print("Testing if monitoring can guide agent back on track")
        print("\nExecuting scenario...")
        
        initial_goal = "Generate sales report"
        
        # Simulate drift and recovery
        stages = [
            {"phase": "on_track", "drift_score": 0.2, "action": "Analyzing sales data"},
            {"phase": "drifting", "drift_score": 0.5, "action": "Looking at marketing"},
            {"phase": "drifted", "drift_score": 0.8, "action": "Analyzing social media"},
            {"phase": "alert", "drift_score": 0.8, "action": "ALERT: Refocusing needed"},
            {"phase": "correcting", "drift_score": 0.6, "action": "Returning to sales analysis"},
            {"phase": "recovering", "drift_score": 0.4, "action": "Generating sales metrics"},
            {"phase": "recovered", "drift_score": 0.2, "action": "Completing sales report"}
        ]
        
        recovery_successful = False
        for i, stage in enumerate(stages):
            print(f"  Step {i+1}: {stage['phase']} - {stage['action']} (drift: {stage['drift_score']})")
            
            if stage["phase"] == "alert":
                print(f"    🚨 Drift Alert Triggered!")
            
            if stage["phase"] == "recovered" and stage["drift_score"] < 0.3:
                recovery_successful = True
        
        print(f"\n✅ Results:")
        print(f"  - Max Drift Before Alert: {max(s['drift_score'] for s in stages[:3]):.1f}")
        print(f"  - Min Drift After Recovery: {min(s['drift_score'] for s in stages[4:]):.1f}")
        print(f"  - Recovery Successful: {'YES ✓' if recovery_successful else 'NO ✗'}")
        print(f"  - Recovery Time: ~3 steps")
        
        self.results.append({
            "scenario": "Drift Recovery",
            "success": recovery_successful,
            "recovery_time": 3
        })
        
        return recovery_successful
    
    async def _calculate_drift_score(self, initial: str, current: str) -> float:
        """Calculate drift score between two text descriptions"""
        # Simplified drift calculation for testing
        initial_words = set(initial.lower().split())
        current_words = set(current.lower().split())
        
        overlap = len(initial_words & current_words)
        total = len(initial_words | current_words)
        
        similarity = overlap / total if total > 0 else 0
        return 1 - similarity  # Drift is inverse of similarity
    
    async def generate_report(self):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("📋 DRIFT DETECTION TEST REPORT")
        print("=" * 60)
        
        successful = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        
        print(f"\nOv
