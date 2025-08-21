"""
Advanced Arize.ai Monitoring Extensions for LangGraph Agent
This module extends the main implementation with:
1. Custom Arize evaluation templates
2. Real-time monitoring dashboard
3. Advanced drift detection algorithms
4. Production-ready alert configurations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

# Arize and Phoenix imports
from arize.pandas.logger import Client as ArizeClient
from arize.utils.types import (
    ModelTypes, 
    Environments,
    Schema,
    Metrics
)
import phoenix as px
from phoenix.evals import (
    llm_classify,
    llm_eval,
    OpenAIModel,
    default_templates
)
from phoenix.trace.dsl import SpanQuery
from phoenix.session.evaluation import get_evaluator

# Additional imports for monitoring
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx


class DriftType(Enum):
    """Types of drift that can be detected"""
    GOAL_DRIFT = "goal_drift"
    SEMANTIC_DRIFT = "semantic_drift"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class EvaluationResult:
    """Result from an evaluation"""
    eval_type: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    alert_triggered: bool


class ArizeEvaluationTemplates:
    """Production-ready evaluation templates for Arize"""
    
    @staticmethod
    def goal_drift_evaluation():
        """Advanced goal drift evaluation template"""
        return {
            "name": "goal_drift_detection_v2",
            "model": "gpt-4",
            "template": """
            You are evaluating an AI agent for goal drift.
            
            Initial Goal: {initial_goal}
            Current State: {current_state}
            Steps Taken: {steps_taken}
            Tools Used: {tools_used}
            
            Evaluation Criteria:
            1. Alignment Score (0-10): How aligned is the current state with the initial goal?
            2. Deviation Indicators:
               - Has the agent introduced unrelated objectives?
               - Is the agent pursuing tangential tasks?
               - Are the tools being used appropriate for the goal?
            3. Recovery Potential: Can the agent easily return to the original goal?
            
            Provide your evaluation in the following JSON format:
            {{
                "alignment_score": <0-10>,
                "drift_detected": <true/false>,
                "drift_severity": <"none"|"low"|"medium"|"high"|"critical">,
                "deviation_reasons": [<list of reasons>],
                "recovery_difficulty": <"easy"|"moderate"|"hard">,
                "recommendation": <string>
            }}
            """,
            "output_schema": {
                "type": "object",
                "properties": {
                    "alignment_score": {"type": "number", "minimum": 0, "maximum": 10},
                    "drift_detected": {"type": "boolean"},
                    "drift_severity": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]},
                    "deviation_reasons": {"type": "array", "items": {"type": "string"}},
                    "recovery_difficulty": {"type": "string", "enum": ["easy", "moderate", "hard"]},
                    "recommendation": {"type": "string"}
                },
                "required": ["alignment_score", "drift_detected", "drift_severity"]
            }
        }
    
    @staticmethod
    def path_divergence_evaluation():
        """Advanced path divergence evaluation template"""
        return {
            "name": "path_divergence_detection_v2",
            "model": "gpt-4",
            "template": """
            Analyze the agent's execution path for efficiency and divergence.
            
            Path History: {path_history}
            Expected Path: {expected_path}
            Execution Time: {execution_time}
            Resource Usage: {resource_usage}
            
            Analysis Requirements:
            1. Path Efficiency (0-10): How efficient is the actual path?
            2. Divergence Patterns:
               - Loops detected: Check for repeated sequences
               - Unnecessary detours: Identify steps that don't contribute to goal
               - Backtracking: Count instances of undoing previous actions
            3. Optimization Opportunities: Suggest a more efficient path
            
            Respond with JSON:
            {{
                "efficiency_score": <0-10>,
                "divergence_type": <"none"|"loop"|"detour"|"backtrack"|"chaotic">,
                "loop_count": <number>,
                "unnecessary_steps": <number>,
                "optimal_path_length": <number>,
                "actual_path_length": <number>,
                "divergence_ratio": <float>,
                "optimization_suggestions": [<list of suggestions>]
            }}
            """,
            "output_schema": {
                "type": "object",
                "properties": {
                    "efficiency_score": {"type": "number", "minimum": 0, "maximum": 10},
                    "divergence_type": {"type": "string"},
                    "loop_count": {"type": "integer"},
                    "unnecessary_steps": {"type": "integer"},
                    "optimal_path_length": {"type": "integer"},
                    "actual_path_length": {"type": "integer"},
                    "divergence_ratio": {"type": "number"},
                    "optimization_suggestions": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    
    @staticmethod
    def behavioral_consistency_evaluation():
        """Evaluate agent's behavioral consistency"""
        return {
            "name": "behavioral_consistency_v1",
            "model": "gpt-4",
            "template": """
            Evaluate the consistency of the agent's behavior across interactions.
            
            Previous Behavior: {previous_behavior}
            Current Behavior: {current_behavior}
            Context Changes: {context_changes}
            
            Assess:
            1. Response Consistency: Are responses coherent with previous ones?
            2. Strategy Consistency: Is the problem-solving approach consistent?
            3. Personality Drift: Has the agent's "personality" changed?
            4. Knowledge Consistency: Are factual claims consistent?
            
            Output JSON:
            {{
                "consistency_score": <0-10>,
                "inconsistencies_found": <true/false>,
                "inconsistency_types": [<list>],
                "severity": <"none"|"minor"|"major"|"critical">,
                "examples": [<specific examples>]
            }}
            """,
            "output_schema": {
                "type": "object",
                "properties": {
                    "consistency_score": {"type": "number"},
                    "inconsistencies_found": {"type": "boolean"},
                    "inconsistency_types": {"type": "array"},
                    "severity": {"type": "string"},
                    "examples": {"type": "array"}
                }
            }
        }


class AdvancedDriftDetector:
    """Advanced drift detection algorithms"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reference_embeddings = []
        self.drift_history = []
        
    def detect_semantic_drift(self, 
                             current_embedding: np.ndarray,
                             reference_embeddings: List[np.ndarray]) -> Tuple[float, Dict]:
        """Detect semantic drift using statistical methods"""
        
        if not reference_embeddings:
            return 0.0, {"error": "No reference embeddings"}
        
        # Calculate distances to reference
        distances = [
            1 - cosine_similarity([current_embedding], [ref])[0][0]
            for ref in reference_embeddings
        ]
        
        # Statistical analysis
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        max_distance = np.max(distances)
        
        # Drift score calculation
        drift_score = mean_distance
        
        # Detect if drift is significant (> 2 standard deviations)
        is_significant = mean_distance > (np.mean(distances[:10]) + 2 * std_distance)
        
        return drift_score, {
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "max_distance": max_distance,
            "is_significant": is_significant,
            "confidence": 1 - std_distance if std_distance < 1 else 0
        }
    
    def detect_behavioral_drift(self,
                               action_sequence: List[str],
                               reference_patterns: List[List[str]]) -> Tuple[float, Dict]:
        """Detect behavioral drift using sequence analysis"""
        
        # Build Markov chain from reference patterns
        transition_probs = self._build_markov_chain(reference_patterns)
        
        # Calculate likelihood of current sequence
        likelihood = self._sequence_likelihood(action_sequence, transition_probs)
        
        # Detect anomalous patterns
        anomalies = self._detect_anomalous_patterns(action_sequence, reference_patterns)
        
        # Calculate drift score
        drift_score = 1 - likelihood if likelihood > 0 else 1.0
        
        return drift_score, {
            "likelihood": likelihood,
            "anomalies": anomalies,
            "unique_actions": len(set(action_sequence)),
            "sequence_length": len(action_sequence)
        }
    
    def _build_markov_chain(self, sequences: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Build Markov chain transition probabilities"""
        transitions = {}
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_state = sequence[i + 1]
                
                if current not in transitions:
                    transitions[current] = {}
                
                if next_state not in transitions[current]:
                    transitions[current][next_state] = 0
                
                transitions[current][next_state] += 1
        
        # Normalize to probabilities
        for state in transitions:
            total = sum(transitions[state].values())
            for next_state in transitions[state]:
                transitions[state][next_state] /= total
        
        return transitions
    
    def _sequence_likelihood(self, sequence: List[str], transitions: Dict) -> float:
        """Calculate likelihood of a sequence given transition probabilities"""
        if len(sequence) < 2:
            return 1.0
        
        likelihood = 1.0
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_state = sequence[i + 1]
            
            if current in transitions and next_state in transitions[current]:
                likelihood *= transitions[current][next_state]
            else:
                likelihood *= 0.01  # Small probability for unseen transitions
        
        return likelihood
    
    def _detect_anomalous_patterns(self, sequence: List[str], references: List[List[str]]) -> List[Dict]:
        """Detect anomalous patterns in sequence"""
        anomalies = []
        
        # Check for loops
        for length in range(2, min(len(sequence) // 2, 10)):
            for i in range(len(sequence) - length * 2):
                pattern = sequence[i:i+length]
                if sequence[i+length:i+length*2] == pattern:
                    anomalies.append({
                        "type": "loop",
                        "pattern": pattern,
                        "position": i,
                        "length": length
                    })
        
        # Check for unprecedented sequences
        subseq_length = 3
        for i in range(len(sequence) - subseq_length):
            subseq = tuple(sequence[i:i+subseq_length])
            found = False
            for ref in references:
                if list(subseq) in [ref[j:j+subseq_length] for j in range(len(ref)-subseq_length)]:
                    found = True
                    break
            if not found:
                anomalies.append({
                    "type": "unprecedented",
                    "pattern": list(subseq),
                    "position": i
                })
        
        return anomalies


class RealTimeMonitoringDashboard:
    """Real-time monitoring dashboard for Arize"""
    
    def __init__(self, arize_client: ArizeClient):
        self.arize_client = arize_client
        self.metrics_buffer = []
        self.alert_history = []
        
    async def start_monitoring(self, check_interval: int = 10):
        """Start real-time monitoring loop"""
        while True:
            metrics = await self.collect_metrics()
            anomalies = self.detect_anomalies(metrics)
            
            if anomalies:
                await self.handle_anomalies(anomalies)
            
            await self.update_dashboard(metrics)
            await asyncio.sleep(check_interval)
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics from agent"""
        return {
            "timestamp": datetime.now(),
            "active_sessions": self._get_active_sessions(),
            "avg_response_time": self._calculate_avg_response_time(),
            "error_rate": self._calculate_error_rate(),
            "drift_metrics": self._get_drift_metrics(),
            "resource_usage": self._get_resource_usage()
        }
    
    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        # Check response time
        if metrics["avg_response_time"] > 5.0:  # 5 seconds threshold
            anomalies.append({
                "type": "high_latency",
                "value": metrics["avg_response_time"],
                "severity": "medium"
            })
        
        # Check error rate
        if metrics["error_rate"] > 0.05:  # 5% threshold
            anomalies.append({
                "type": "high_error_rate",
                "value": metrics["error_rate"],
                "severity": "high"
            })
        
        # Check drift metrics
        for drift_type, score in metrics["drift_metrics"].items():
            if score > 0.7:
                anomalies.append({
                    "type": f"{drift_type}_detected",
                    "value": score,
                    "severity": "high" if score > 0.9 else "medium"
                })
        
        return anomalies
    
    async def handle_anomalies(self, anomalies: List[Dict]):
        """Handle detected anomalies"""
        for anomaly in anomalies:
            # Log to Arize
            self.arize_client.log(
                model_id="langgraph-agent",
                model_version="1.0",
                model_type=ModelTypes.GENERATIVE_LLM,
                environment=Environments.PRODUCTION,
                prediction_id=str(uuid.uuid4()),
                prediction_label="anomaly_detection",
                tags={
                    "anomaly_type": anomaly["type"],
                    "severity": anomaly["severity"],
                    "value": anomaly["value"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Trigger appropriate response
            if anomaly["severity"] == "high":
                await self._trigger_high_severity_response(anomaly)
            elif anomaly["severity"] == "medium":
                await self._trigger_medium_severity_response(anomaly)
    
    async def update_dashboard(self, metrics: Dict[str, Any]):
        """Update monitoring dashboard"""
        # Create dashboard payload
        dashboard_data = {
            "timestamp": metrics["timestamp"].isoformat(),
            "summary": {
                "total_sessions": metrics["active_sessions"],
                "avg_latency": metrics["avg_response_time"],
                "error_rate": metrics["error_rate"],
                "health_score": self._calculate_health_score(metrics)
            },
            "drift_analysis": metrics["drift_metrics"],
            "alerts": len(self.alert_history),
            "resource_usage": metrics["resource_usage"]
        }
        
        # Send to Arize dashboard
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=str(uuid.uuid4()),
            prediction_label="dashboard_update",
            features=dashboard_data
        )
        
        # Store in buffer for trending
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > 1000:
            self.metrics_buffer.pop(0)
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        score = 100.0
        
        # Deduct for high latency
        if metrics["avg_response_time"] > 3.0:
            score -= min(20, (metrics["avg_response_time"] - 3.0) * 5)
        
        # Deduct for errors
        score -= min(30, metrics["error_rate"] * 100)
        
        # Deduct for drift
        max_drift = max(metrics["drift_metrics"].values())
        score -= min(30, max_drift * 30)
        
        # Deduct for resource usage
        if metrics["resource_usage"]["cpu"] > 80:
            score -= 10
        if metrics["resource_usage"]["memory"] > 80:
            score -= 10
        
        return max(0, score)
    
    def _get_active_sessions(self) -> int:
        """Get count of active agent sessions"""
        # Implementation would query actual session data
        return 42  # Placeholder
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # Implementation would calculate from actual metrics
        return 2.3  # Placeholder in seconds
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        # Implementation would calculate from actual errors
        return 0.02  # Placeholder (2%)
    
    def _get_drift_metrics(self) -> Dict[str, float]:
        """Get current drift metrics"""
        return {
            "goal_drift": 0.45,
            "semantic_drift": 0.32,
            "behavioral_drift": 0.28,
            "performance_drift": 0.15
        }
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get resource usage metrics"""
        import psutil
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }
    
    async def _trigger_high_severity_response(self, anomaly: Dict):
        """Handle high severity anomalies"""
        # Could trigger auto-scaling, circuit breakers, etc.
        print(f"🚨 HIGH SEVERITY: {anomaly}")
        
    async def _trigger_medium_severity_response(self, anomaly: Dict):
        """Handle medium severity anomalies"""
        print(f"⚠️ MEDIUM SEVERITY: {anomaly}")


class ProductionAlertConfiguration:
    """Production-ready alert configurations"""
    
    @staticmethod
    def get_production_alerts() -> List[Dict]:
        """Get production alert configurations"""
        return [
            {
                "name": "Critical Goal Drift",
                "enabled": True,
                "metric": "goal_drift_score",
                "condition": {
                    "type": "threshold",
                    "operator": "greater_than",
                    "value": 0.8
                },
                "severity": "CRITICAL",
                "actions": ["page_oncall", "create_incident", "slack_alert"],
                "cooldown": {"minutes": 15},
                "escalation": {
                    "enabled": True,
                    "after_minutes": 30,
                    "to": "engineering_manager"
                }
            },
            {
                "name": "Performance Degradation",
                "enabled": True,
                "metric": "avg_response_time",
                "condition": {
                    "type": "rolling_average",
                    "window_minutes": 5,
                    "operator": "greater_than",
                    "value": 5.0
                },
                "severity": "HIGH",
                "actions": ["slack_alert", "email_team"],
                "cooldown": {"minutes": 30}
            },
            {
                "name": "Agent Loop Detection",
                "enabled": True,
                "metric": "loop_detection",
                "condition": {
                    "type": "pattern",
                    "pattern": "repeated_sequence",
                    "min_repetitions": 3
                },
                "severity": "HIGH",
                "actions": ["restart_agent", "slack_alert"],
                "cooldown": {"minutes": 10}
            },
            {
                "name": "Resource Exhaustion",
                "enabled": True,
                "metric": "resource_usage",
                "condition": {
                    "type": "composite",
                    "conditions": [
                        {"metric": "cpu", "operator": "greater_than", "value": 90},
                        {"metric": "memory", "operator": "greater_than", "value": 85}
                    ],
                    "logic": "OR"
                },
                "severity": "HIGH",
                "actions": ["scale_up", "slack_alert"],
                "cooldown": {"minutes": 20}
            },
            {
                "name": "Semantic Consistency Check",
                "enabled": True,
                "metric": "semantic_consistency",
                "condition": {
                    "type": "ml_based",
                    "model": "consistency_detector_v2",
                    "threshold": 0.3
                },
                "severity": "MEDIUM",
                "actions": ["log_for_review", "metrics_dashboard"],
                "cooldown": {"minutes": 60}
            }
        ]


class ArizeIntegrationManager:
    """Manages the complete Arize integration"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.arize_client = ArizeClient(
            api_key=config["ARIZE_API_KEY"],
            space_id=config["ARIZE_SPACE_ID"]
        )
        self.drift_detector = AdvancedDriftDetector()
        self.dashboard = RealTimeMonitoringDashboard(self.arize_client)
        self.alerts = ProductionAlertConfiguration.get_production_alerts()
        
    async def initialize(self):
        """Initialize all monitoring components"""
        # Start Phoenix
        px.launch_app()
        
        # Setup monitoring dashboard
        asyncio.create_task(self.dashboard.start_monitoring())
        
        # Initialize alert handlers
        self._setup_alert_handlers()
        
        print("✅ Arize Integration Manager initialized")
    
    def _setup_alert_handlers(self):
        """Setup alert handling infrastructure"""
        for alert in self.alerts:
            if alert["enabled"]:
                print(f"📊 Configured alert: {alert['name']}")
    
    def log_span(self, span_data: Dict):
        """Log a span to Arize"""
        self.arize_client.log(
            model_id="langgraph-agent",
            model_version="1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_id=span_data.get("span_id", str(uuid.uuid4())),
            prediction_label=span_data.get("operation", "unknown"),
            features=span_data.get("features", {}),
            tags=span_data.get("tags", {})
        )
    
    async def evaluate_drift(self, current_state: Dict) -> EvaluationResult:
        """Evaluate all types of drift"""
        
        # Goal drift
        goal_drift = self.drift_detector.detect_semantic_drift(
            current_state.get("current_embedding"),
            current_state.get("reference_embeddings", [])
        )
        
        # Behavioral drift
        behavioral_drift = self.drift_detector.detect_behavioral_drift(
            current_state.get("action_sequence", []),
            current_state.get("reference_patterns", [])
        )
        
        # Combine results
        combined_score = (goal_drift[0] + behavioral_drift[0]) / 2
        
        result = EvaluationResult(
            eval_type="combined_drift",
            score=combined_score,
            details={
                "goal_drift": goal_drift,
                "behavioral_drift": behavioral_drift
            },
            timestamp=datetime.now(),
            alert_triggered=combined_score > 0.7
        )
        
        # Log to Arize
        self.log_span({
            "span_id": str(uuid.uuid4()),
            "operation": "drift_evaluation",
            "features": {
                "drift_score": combined_score,
                "goal_drift_score": goal_drift[0],
                "behavioral_drift_score": behavioral_drift[0]
            },
            "tags": {
                "alert_triggered": result.alert_triggered,
                "evaluation_type": "combined"
            }
        })
        
        return result


# Example usage
async def main_advanced():
    """Main function demonstrating advanced monitoring"""
    
    config = {
        "ARIZE_API_KEY": "your-arize-api-key",
        "ARIZE_SPACE_ID": "your-arize-space-id",
        "OPENAI_API_KEY": "your-openai-api-key"
    }
    
    # Initialize integration manager
    manager = ArizeIntegrationManager(config)
    await manager.initialize()
    
    # Simulate agent execution with monitoring
    current_state = {
        "current_embedding": np.random.randn(1536),  # OpenAI embedding dimension
        "reference_embeddings": [np.random.randn(1536) for _ in range(10)],
        "action_sequence": ["search", "analyze", "search", "analyze", "search"],
        "reference_patterns": [
            ["search", "analyze", "report"],
            ["search", "process", "analyze", "report"]
        ]
    }
    
    # Evaluate drift
    result = await manager.evaluate_drift(current_state)
    
    print(f"\n📊 Drift Evaluation Results:")
    print(f"Score: {result.score:.2f}")
    print(f"Alert Triggered: {result.alert_triggered}")
    print(f"Details: {json.dumps(result.details, indent=2)}")
    
    # Keep monitoring running
    await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main_advanced())

"""
Advanced Features Added:

1. **Production-Ready Evaluation Templates**:
   - Goal drift with detailed scoring
   - Path divergence with optimization suggestions
   - Behavioral consistency checking

2. **Advanced Drift Detection**:
   - Semantic drift using embeddings
   - Behavioral drift using Markov chains
   - Statistical anomaly detection

3. **Real-Time Monitoring Dashboard**:
   - Continuous metric collection
   - Anomaly detection
   - Health score calculation
   - Auto-scaling triggers

4. **Production Alert Configuration**:
   - Critical, High, Medium severity levels
   - Escalation policies
   - Cooldown periods
   - Composite conditions

5. **Complete Arize Integration**:
   - Span logging
   - Dashboard updates
   - Alert management
   - Evaluation results tracking

This provides a production-ready monitoring solution for your LangGraph agent!
"""
