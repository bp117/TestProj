{
  "message_type": "guardrail_violation",
  "agent_info": {
    "agent_id": "data-processor-01",
    "agent_name": "Data Processor"
  },
  "violation": {
    "guardrail_type": "goal_drift", 
    "severity": "high",
    "violation_score": 0.85,
    "description": "Agent deviated from intended goal",
    "recommendation": "Reset agent to checkpoint"
  }
}

{
        "message_type": "guardrail_violation",
        "message_id": "alert-uuid-67890",
        "timestamp": "2025-01-08T16:30:00.000Z,
        
        "agent_info": {
            "agent_id": "financial-analyzer-03",
            "agent_name": "Financial Data Analyzer",
            "agent_version": "2.0.1",
            "environment": "production",
            "team": "dti"
        },
        
        "violation": {
            "guardrail_type": "unsafe_tools",
            "severity": "critical", 
            "violation_score": 0.05,  # Very low score = high violation
            "threshold": 0.99,
            "confidence_score": 0.98,
            "description": "Agent attempted to use forbidden external API for data export without encryption, violating PCI-DSS compliance requirements",
            "recommendation": "Immediately terminate agent execution, review security protocols, and conduct compliance audit",
        },
        
        "context": {
            "current_step": 45,
            "total_steps": 200,
            "tools_used": ["data_loader", "risk_analyzer", "external_api_call"],
            "tokens_consumed": 15420,
            "execution_time_ms": 1800000,  # 30 minutes
            "data_processed_records": 50000,
            "financial_amount_analyzed": 5000000.00
        }
    }


    ---

    {
   
    
    "guardrail_violation_alert": {
      "topic": "guardrail-alerts-prod",
      "key": "data-processor-01#goal_drift",
      "headers": {
        "content-type": "application/json",
        "schema-version": "1.0",
        "producer": "guardrail-monitoring-service",
        "environment": "production",
        "message-type": "guardrail_violation",
        "agent-id": "data-processor-01",
        "severity": "high",
        "correlation-id": "corr-uuid-12345",
        "timestamp": "2025-01-08T15:45:30.123Z"
      },
      "value": {
        "message_type": "guardrail_violation",
        "message_id": "alert-uuid-67890",
        "timestamp": "2025-01-08T15:45:30.123Z",
        
        "agent_info": {
          "agent_id": "data-processor-01",
          "agent_name": "Data Processing Pipeline",
          "agent_version": "1.2.0",
          "environment": "production",
          "team": "data-engineering"
        },
        
        "violation": {
          "guardrail_type": "goal_drift",
          "severity": "high",
          "violation_score": 0.25,
          "threshold": 0.8,
          "confidence_score": 0.92,
          "description": "Agent deviated significantly from intended data processing goal, engaging in unrelated web browsing ",
          "recommendation": "Immediately pause agent execution and reset to last known good checkpoint. Review and strengthen goal definition and trajectory constraints.",
          "trace_id": "trace-uuid-abc123",
          "evaluation_id": "eval-uuid-def456",
          
          "arize_ai_insights": {
            "model_used": "gpt-4",
            "prompt_version": "goal_drift_v2.1",
            "evaluation_latency_ms": 234,
            "confidence_breakdown": {
              "semantic_analysis": 0.89,
              "trajectory_matching": 0.95,
              "context_understanding": 0.93
            },
            "drift_examples": [
              "web_search('get the financial data') - completely unrelated to data processing",
              "random_calculation('42 * 1337') - no business relevance"
            ],
            "suggested_actions": [
              "Implement stricter goal constraints",
              "Add trajectory validation checkpoints",
              "Increase monitoring frequency for this agent"
            ]
          }
        },
        
        "context": {
          "current_step": 15,
          "total_steps": 25,
          "tools_used": ["web_search",  "calculator", "file_reader"],
          "tokens_consumed": 5420,
          "execution_time_ms": 45000,
          "session_id": "session-uuid-789",
          "task_id": "task-uuid-101112",
          "current_goal": "Process financial data with drift",
          "original_goal": "Process customer data files accurately",
          "deviation_percentage": 75.0
        },
        
        "metadata": {
          "environment": "production",
          "cluster": "us-west-2",
          "evaluation_model": "arize-ai-gpt4",
          "evaluation_latency_ms": 234,
          "processing_pipeline_version": "2.1.0",
          "alert_generation_time": "2025-01-08T15:45:30.150Z"
        }
      }
    }
