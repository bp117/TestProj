# 📊 AI Agent Guardrail Framework - Complete Sequence Diagrams

## 🎯 System Overview Architecture

```mermaid
graph TB
    subgraph "AI Agents Layer"
        A1[Data Processor Agent]
        A2[ML Trainer Agent] 
        A3[Custom Agents]
    end
    
    subgraph "Trace Collection"
        OE[OTEL Exporter<br/>5-sec intervals]
        S3[S3 Storage<br/>HPOS Format]
    end
    
    subgraph "Core Processing Engine"
        MS[Monitoring Service<br/>Trace Processor]
        AR[Agent Registry<br/>Redis Cache]
    end
    
    subgraph "🔥 Arize AI Evaluation Engine"
        GE[Guardrail Engine<br/>LLM-as-Judge]
        AI[Arize AI Evals<br/>Phoenix Platform]
        
        subgraph "Guardrail Types"
            GD[Goal Drift<br/>Arize Analysis]
            SB[Step Budgeting<br/>Arize Analysis] 
            UT[Unsafe Tools<br/>Arize Analysis]
            LD[Loop Detection<br/>Arize Analysis]
        end
    end
    
    subgraph "Event Streaming (Simplified)"
        MK[Managed Kafka<br/>or Kafka in Docker]
        KP[Kafka Producer<br/>Alerts & Metrics]
        KC[Kafka Consumer<br/>Dashboard Updates]
    end
    
    subgraph "Frontend & APIs"
        FB[FastAPI Backend<br/>REST + WebSocket]
        RD[React Dashboard<br/>Real-time UI]
        WS[WebSocket Server<br/>Live Updates]
    end
    
    subgraph "Infrastructure (Minimal)"
        R[Redis Cache<br/>Agent Registry]
        LS[LocalStack S3<br/>Development Only]
    end
    
    %% Flow connections
    A1 --> OE
    A2 --> OE
    A3 --> OE
    OE --> S3
    S3 --> MS
    MS --> AR
    MS --> GE
    
    %% Arize AI Integration (Highlighted)
    GE --> AI
    AI --> GD
    AI --> SB
    AI --> UT
    AI --> LD
    GD --> AI
    SB --> AI
    UT --> AI
    LD --> AI
    
    %% Event flow
    GE --> KP
    KP --> MK
    MK --> KC
    KC --> WS
    WS --> RD
    
    %% Data storage
    AR --> R
    FB --> RD
    
    %% Styling
    classDef arizeAI fill:#ff6b6b,stroke:#d63031,stroke-width:3px,color:#fff
    classDef core fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    classDef infra fill:#a29bfe,stroke:#6c5ce7,stroke-width:2px
    
    class AI,GE,GD,SB,UT,LD arizeAI
    class MS,AR,OE core
    class R,MK,S3 infra
```

---

## 1. 🔄 Trace Collection and Storage Flow

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant OTEL as OTEL Exporter
    participant Processor as Span Processor
    participant S3 as S3 Storage
    participant Monitor as Monitoring Service
    
    Note over Agent,Monitor: Every 5 seconds cycle
    
    %% Agent execution with tracing
    Agent->>Agent: Start task execution
    Agent->>OTEL: Generate OpenTelemetry spans
    
    Note over Agent: agent.id, session.id, task.id<br/>operation_name, attributes
    
    OTEL->>Processor: on_start(span)
    Processor->>Processor: Extract agent context<br/>Initialize trace object
    
    Agent->>Agent: Execute operations<br/>(tools, steps, decisions)
    
    loop For each operation
        Agent->>OTEL: Create span with attributes
        OTEL->>Processor: Track span data
        Processor->>Processor: Update agent state<br/>(steps, tokens, tools used)
    end
    
    Agent->>OTEL: Complete task (trace.complete=true)
    OTEL->>Processor: on_end(span)
    Processor->>Processor: Calculate durations<br/>Finalize trace data
    
    %% Trace collection and export
    Note over OTEL,S3: 5-second export interval
    
    Processor->>OTEL: Add completed trace to buffer
    OTEL->>OTEL: Transform to HPOS format
    
    OTEL->>S3: Upload trace file<br/>(/hpos/{agent_id}/{timestamp}.json)
    
    Note over S3: Partitioned by agent and time<br/>/hpos/agent-001/2025/01/08/14/trace.json
    
    S3->>Monitor: S3 Event Notification<br/>(new file available)
    Monitor->>S3: Fetch new trace files<br/>(discovery loop)
```

### 🔍 **Flow Explanation: Trace Collection**

**Phase 1: Agent Execution (Real-time)**
1. **Agent starts task** - Begins execution with specific goal
2. **OpenTelemetry instrumentation** - Every operation creates spans
3. **Span attributes capture**:
   - `agent.id`, `session.id`, `task.id`
   - `operation_name` (e.g., "tool_web_search")
   - `tool.name`, `tool.input`, `tool.output`
   - `agent.step`, `agent.goal`, `tokens.used`

**Phase 2: Trace Processing (Continuous)**
1. **Span processor** aggregates spans into complete traces
2. **Agent state tracking** - Steps, tokens, tools used
3. **Trace completion detection** - Root span ends or `trace.complete` flag

**Phase 3: Export to Storage (Every 5 seconds)**
1. **HPOS transformation** - Convert to structured format
2. **S3 partitioning** - Organized by agent ID and timestamp
3. **Event notification** - S3 triggers monitoring service

**Key Benefits:**
- ✅ **Non-blocking collection** - Doesn't slow down agents
- ✅ **Structured storage** - Easy querying and analysis
- ✅ **Scalable partitioning** - Supports many agents
- ✅ **Real-time processing** - 5-second latency

---

## 2. 🛡️ Guardrail Evaluation Flow

```mermaid
sequenceDiagram
    participant MS as Monitoring Service
    participant S3 as S3 Storage
    participant AR as Agent Registry
    participant GE as Guardrail Engine
    participant GD as Goal Drift Guardrail
    participant SB as Step Budgeting Guardrail
    participant UT as Unsafe Tools Guardrail
    participant LD as Loop Detection Guardrail
    participant AI as Arize AI Evals
    participant KP as Kafka Producer
    
    %% Trace processing trigger
    MS->>S3: Poll for new trace files<br/>(discovery loop)
    S3->>MS: Return new trace files
    
    loop For each new trace file
        MS->>S3: Download trace file
        S3->>MS: Return trace data (HPOS format)
        
        MS->>MS: Parse trace to AgentTrace object
        MS->>AR: Get agent configuration
        AR->>MS: Return agent + guardrail config
        
        %% Evaluate each enabled guardrail
        Note over MS,LD: Parallel evaluation of enabled guardrails
        
        par Goal Drift Evaluation
            alt Goal drift enabled
                MS->>GD: evaluate_trace(trace, agent)
                GD->>GD: Extract context<br/>(goal, trajectory, steps)
                GD->>AI: LLM-as-Judge evaluation<br/>(semantic analysis)
                AI->>GD: Violation score + explanation
                GD->>GD: Determine severity<br/>(threshold comparison)
                GD->>MS: GuardrailEvaluationResult
            end
        and Step Budgeting Evaluation
            alt Step budgeting enabled
                MS->>SB: evaluate_trace(trace, agent)
                SB->>SB: Extract resource usage<br/>(steps, tokens, time)
                SB->>AI: LLM-as-Judge evaluation<br/>(efficiency analysis)
                AI->>SB: Budget violation score
                SB->>SB: Calculate utilization<br/>(usage vs limits)
                SB->>MS: GuardrailEvaluationResult
            end
        and Unsafe Tools Evaluation
            alt Unsafe tools enabled
                MS->>UT: evaluate_trace(trace, agent)
                UT->>UT: Extract tool usage<br/>(tools, combinations, patterns)
                UT->>AI: LLM-as-Judge evaluation<br/>(security analysis)
                AI->>UT: Safety violation score
                UT->>UT: Check forbidden tools<br/>(blacklist matching)
                UT->>MS: GuardrailEvaluationResult
            end
        and Loop Detection Evaluation
            alt Loop detection enabled
                MS->>LD: evaluate_trace(trace, agent)
                LD->>LD: Extract execution patterns<br/>(sequences, repetitions)
                LD->>AI: LLM-as-Judge evaluation<br/>(pattern analysis)
                AI->>LD: Loop detection score
                LD->>LD: Analyze similarity<br/>(input/output matching)
                LD->>MS: GuardrailEvaluationResult
            end
        end
        
        %% Process evaluation results
        MS->>MS: Aggregate evaluation results
        
        loop For each violation detected
            MS->>MS: Create GuardrailViolation object<br/>(with context and metadata)
            MS->>KP: Send violation alert<br/>(Kafka message)
            MS->>MS: Update agent metrics<br/>(violation counts, stats)
        end
        
        MS->>AR: Update agent metrics cache
    end
    
    Note over MS: Continuous processing<br/>Batch size: 10 traces
```

### 🔍 **Flow Explanation: Guardrail Evaluation**

**Phase 1: Trace Discovery (Continuous)**
1. **S3 polling** - Monitoring service discovers new trace files
2. **Trace parsing** - Convert HPOS format to AgentTrace objects
3. **Agent configuration** - Fetch guardrail settings from registry

**Phase 2: Parallel Evaluation (Per Trace)**
1. **Guardrail selection** - Only evaluate enabled guardrails
2. **Context extraction** - Each guardrail extracts relevant data:
   - **Goal Drift**: Current goal, step sequence, trajectory
   - **Step Budgeting**: Resource consumption, efficiency metrics
   - **Unsafe Tools**: Tool usage patterns, security violations
   - **Loop Detection**: Execution patterns, repetition analysis

**Phase 3: LLM-as-Judge Analysis**
1. **Prompt generation** - Create evaluation prompts with context
2. **Arize AI Evals** - Advanced AI-powered evaluation
3. **Fallback logic** - Rule-based evaluation if LLM fails
4. **Confidence scoring** - 0-1 confidence in violation detection

**Phase 4: Violation Processing**
1. **Severity determination** - Critical/High/Medium/Low based on scores
2. **Context enrichment** - Add execution metadata and recommendations
3. **Alert generation** - Create Kafka messages for violations
4. **Metrics updates** - Update agent statistics and health scores

**Key Features:**
- 🔄 **Parallel processing** - Multiple guardrails evaluate simultaneously
- 🤖 **AI-powered evaluation** - LLM-as-Judge for sophisticated analysis
- 🛡️ **Fallback mechanisms** - Rule-based backup when AI fails
- 📊 **Rich context** - Detailed violation explanations and recommendations

---

## 3. 📡 Real-time Dashboard Updates Flow

```mermaid
sequenceDiagram
    participant UI as React Dashboard
    participant WS as WebSocket Server
    participant KC as Kafka Consumer<br/>(dashboard-updater)
    participant K as Kafka Topics
    participant KP as Kafka Producer
    participant MS as Monitoring Service
    
    %% Initial connection
    UI->>WS: Connect to WebSocket<br/>(ws://localhost:8000/ws/dashboard)
    WS->>WS: Accept connection<br/>Add to active connections
    WS->>UI: Connection confirmed<br/>(connection_established message)
    
    %% Kafka consumer setup
    KC->>K: Subscribe to topics<br/>(alerts, metrics, status)
    
    Note over KC,K: Consumer group: dashboard-updater<br/>Topics: guardrail-alerts, agent-metrics, agent-status
    
    %% Real-time event flow
    loop Continuous Real-time Updates
        
        %% Violation alert flow
        MS->>KP: Guardrail violation detected
        KP->>K: Publish to guardrail-alerts topic
        
        Note over K: Message includes:<br/>agent_id, violation_type, severity,<br/>description, recommendation
        
        K->>KC: Consume violation alert
        KC->>KC: Process alert message<br/>(_handle_violation_alert)
        KC->>WS: Send dashboard update
        
        Note over WS: Broadcast to all connected clients
        
        WS->>UI: Real-time violation alert<br/>(type: violation_alert)
        UI->>UI: Update violation counter<br/>Show notification toast<br/>Add to alerts list
        
        %% Metrics update flow
        par Metrics Updates
            MS->>KP: Agent metrics update
            KP->>K: Publish to agent-metrics topic
            K->>KC: Consume metrics update
            KC->>KC: Process metrics<br/>(_handle_metrics_update)
            KC->>WS: Send metrics data
            WS->>UI: Real-time metrics<br/>(type: metrics_update)
            UI->>UI: Update performance charts<br/>Refresh KPI cards
        and Status Change Flow
            MS->>KP: Agent status change
            KP->>K: Publish to agent-status topic
            K->>KC: Consume status change
            KC->>KC: Process status<br/>(_handle_status_change)
            KC->>WS: Send status update
            WS->>UI: Real-time status<br/>(type: status_change)
            UI->>UI: Update agent status badge<br/>Refresh agent list
        end
        
    end
    
    %% Client-initiated interactions
    UI->>WS: Send client message<br/>(ping, subscribe, get_status)
    WS->>WS: Handle client message<br/>(handle_client_message)
    WS->>UI: Response message<br/>(pong, subscription_confirmed)
    
    %% Connection management
    Note over UI,WS: Connection health monitoring
    
    loop Every 30 seconds
        UI->>WS: Ping message
        WS->>UI: Pong response
        
        alt Connection lost
            UI->>UI: Show "Disconnected" status<br/>Attempt reconnection
            UI->>WS: Reconnect with exponential backoff
        end
    end
    
    %% Cleanup on disconnect
    UI->>WS: Disconnect (user closes browser)
    WS->>WS: Remove from active connections<br/>Cleanup resources
```

### 🔍 **Flow Explanation: Real-time Dashboard Updates**

**Phase 1: Connection Establishment**
1. **WebSocket connection** - React app connects on page load
2. **Connection management** - Server tracks active connections
3. **Initial state** - Dashboard receives current system status

**Phase 2: Event Streaming Setup**
1. **Kafka consumer** - dashboard-updater group subscribes to topics
2. **Topic subscriptions**:
   - `guardrail-alerts` - Violation notifications
   - `agent-metrics` - Performance updates
   - `agent-status` - Agent state changes

**Phase 3: Real-time Event Processing**
1. **Violation alerts**:
   - Monitoring service detects violation → Kafka
   - Consumer processes → WebSocket broadcast
   - Dashboard shows toast notification + updates counters

2. **Metrics updates**:
   - Agent performance metrics → Kafka (every 60 seconds)
   - Dashboard updates charts and KPI cards in real-time

3. **Status changes**:
   - Agent starts/stops/errors → Kafka
   - Dashboard updates status badges and agent list

**Phase 4: Connection Health**
1. **Ping/pong mechanism** - 30-second heartbeat
2. **Reconnection logic** - Exponential backoff on disconnect
3. **Status indicators** - Green/red dot shows connection state

**Real-time Capabilities:**
- ⚡ **Sub-second updates** - Violations appear immediately
- 📊 **Live charts** - Performance metrics update automatically
- 🔄 **Auto-reconnection** - Handles network interruptions
- 🎯 **Targeted updates** - Only relevant data sent to each client

---

## 4. 👥 Agent Registration and Configuration Flow

```mermaid
sequenceDiagram
    participant User as User/System
    participant API as FastAPI Backend
    participant AR as Agent Registry
    participant R as Redis Cache
    participant OE as OTEL Exporter
    participant MS as Monitoring Service
    participant KP as Kafka Producer
    
    %% Agent registration
    User->>API: POST /api/agents/<br/>(agent configuration)
    
    Note over User,API: Agent data includes:<br/>agent_id, name, description,<br/>guardrail configurations
    
    API->>API: Validate agent data<br/>(Pydantic models)
    API->>AR: Check if agent exists
    AR->>R: Query existing agent
    R->>AR: Agent not found
    AR->>API: Proceed with registration
    
    API->>AR: register_agent(agent)
    AR->>AR: Set timestamps<br/>(created_at, updated_at)
    AR->>R: Store agent data<br/>(agent:{agent_id})
    AR->>R: Update sorted set<br/>(agents_by_updated)
    
    Note over R: Redis keys:<br/>agent:data-processor-01<br/>agents_by_updated (sorted set)
    
    AR->>API: Return registered agent
    API->>User: HTTP 201 Created<br/>(agent object)
    
    %% Service registration
    par OTEL Exporter Registration
        AR->>OE: Notify new agent registered
        OE->>OE: register_agent(agent)<br/>Add to active_agents
        OE->>OE: Initialize trace buffer<br/>trace_buffer[agent_id] = []
    and Monitoring Service Setup
        AR->>MS: Notify agent configuration
        MS->>MS: Load guardrail configurations<br/>for new agent
    end
    
    %% Status change notification
    AR->>KP: Send agent status message
    KP->>KP: Create AgentStatusMessage<br/>(type: agent_registration)
    
    Note over KP: Message includes:<br/>agent_id, status: "running",<br/>guardrails enabled
    
    %% Configuration updates
    User->>API: PUT /api/agents/{agent_id}/guardrails<br/>(updated guardrail config)
    API->>AR: update_agent_guardrails(agent_id, config)
    
    AR->>R: Update agent data<br/>(modified guardrails)
    AR->>R: Update timestamp
    AR->>API: Return updated agent
    API->>User: HTTP 200 OK<br/>(updated agent)
    
    %% Propagate configuration changes
    AR->>MS: Configuration changed<br/>(reload guardrail settings)
    MS->>MS: Update agent configuration cache
    AR->>KP: Send configuration update message
    
    %% Status management
    User->>API: PATCH /api/agents/{agent_id}/status?status=paused
    API->>AR: update_agent_status(agent_id, "paused")
    AR->>R: Update agent status
    AR->>API: Return updated agent
    
    %% Service notifications for status change
    AR->>OE: Agent status changed to paused
    OE->>OE: Stop collecting traces<br/>for paused agent
    AR->>MS: Agent paused<br/>(suspend monitoring)
    MS->>MS: Pause guardrail evaluations<br/>for this agent
    AR->>KP: Send status change message
    
    %% Agent health monitoring
    loop Continuous Health Monitoring
        AR->>AR: Calculate agent health scores<br/>(violation rates, uptime)
        AR->>R: Update cached metrics<br/>(agent_metrics:{agent_id})
        
        alt Health degradation detected
            AR->>KP: Send health alert
            AR->>MS: Trigger detailed evaluation
        end
    end
```

### 🔍 **Flow Explanation: Agent Registration and Configuration**

**Phase 1: Agent Registration**
1. **Validation** - Pydantic models ensure data integrity
2. **Uniqueness check** - Prevent duplicate agent IDs
3. **Timestamp management** - Track creation and modification times
4. **Redis storage** - Cached for fast access with sorted sets

**Phase 2: Service Integration**
1. **OTEL registration** - Begin trace collection for new agent
2. **Monitoring setup** - Load guardrail configurations
3. **Buffer initialization** - Prepare trace storage structures

**Phase 3: Configuration Management**
1. **Guardrail updates** - Modify detection rules and thresholds
2. **Real-time propagation** - Changes applied without restart
3. **Version tracking** - Audit trail of configuration changes

**Phase 4: Status Management**
1. **Status transitions** - running ↔ paused ↔ stopped ↔ error
2. **Service coordination** - All services respect status changes
3. **Event notifications** - Kafka messages for status updates

**Registry Features:**
- 🏪 **Centralized configuration** - Single source of truth
- 🔄 **Real-time updates** - No service restarts needed
- 📊 **Health monitoring** - Continuous agent health assessment
- 🎯 **Event-driven** - Configuration changes trigger notifications

---

## 5. 🚨 Violation Detection and Alerting Flow

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant GE as Guardrail Engine
    participant AI as Arize AI Evals
    participant MS as Monitoring Service
    participant KP as Kafka Producer
    participant K as Kafka Topics
    participant KC as Kafka Consumer
    participant WS as WebSocket Server
    participant UI as Dashboard
    participant Slack as External Systems (Slack, Email, PagerDuty)
    
    %% Violation detection
    Agent->>Agent: Execute problematic behavior<br/>(goal drift, unsafe tools, etc.)
    
    Note over Agent: Example: Agent starts browsing<br/>social media instead of processing data
    
    Agent->>GE: Trace contains violation indicators<br/>(via monitoring service)
    
    %% Goal drift detection example
    GE->>GE: Extract execution context<br/>(current vs intended goal)
    GE->>AI: LLM-as-Judge evaluation
    
    Note over AI: Prompt: "Agent intended to process data<br/>but executed: web_search('funny cat videos'),<br/>social_media_check(), random_calculation()"
    
    AI->>GE: Violation detected!<br/>Confidence: 0.92, Severity: HIGH
    
    GE->>MS: GuardrailEvaluationResult<br/>(violation_detected=true)
    
    %% Violation processing
    MS->>MS: Create GuardrailViolation object<br/>with rich context
    
    Note over MS: Violation includes:<br/>- Agent info (ID, name, version)<br/>- Violation details (type, severity)<br/>- Execution context (steps, tools, tokens)<br/>- Recommendations (remediation steps)
    
    MS->>KP: Send violation alert
    
    %% Kafka message creation
    KP->>KP: Create GuardrailAlertMessage
    
    Note over KP: Message key: "{agent_id}#{guardrail_type}"<br/>Headers: severity, correlation-id, timestamp<br/>Partition: Based on agent_id for ordering
    
    KP->>K: Publish to guardrail-alerts topic
    
    %% Multi-consumer processing
    par Dashboard Updates
        K->>KC: Consume alert (dashboard-updater group)
        KC->>KC: Process violation alert<br/>(_handle_violation_alert)
        KC->>WS: Send real-time update
        WS->>UI: violation_alert message
        
        UI->>UI: Show toast notification<br/>"🚨 HIGH: Goal drift detected<br/>in Data Processor Agent"
        UI->>UI: Update metrics dashboard<br/>Increment violation counter<br/>Add to recent alerts list
        
    and External Alerting
        K->>KC: Consume alert (alert-manager group) 
        KC->>KC: Process for external systems<br/>Check severity thresholds
        
        alt Critical or High severity
            KC->>Slack: Send Slack notification<br/>"🚨 Critical violation in agent-001"
            KC->>KC: Send email alert<br/>PagerDuty notification
        end
        
    and Audit Logging
        K->>KC: Consume alert (audit-logger group)
        KC->>KC: Log violation for compliance<br/>Store in audit database
        KC->>KC: Update violation statistics<br/>Generate compliance reports
    end
    
    %% Remediation actions
    UI->>Agent: User triggers remediation<br/>(via dashboard controls)
    
    alt Automatic remediation
        MS->>Agent: Auto-pause agent<br/>(if critical violation)
        Agent->>Agent: Stop current execution<br/>Reset to safe state
    else Manual intervention
        UI->>MS: User initiates agent reset<br/>Change configuration
        MS->>Agent: Apply configuration changes<br/>Resume with new guardrails
    end
    
    %% Follow-up monitoring
    MS->>MS: Increase monitoring frequency<br/>for recently violated agent
    MS->>GE: Enhanced evaluation<br/>(stricter thresholds temporarily)
    
    %% Resolution tracking
    UI->>MS: Mark violation as resolved<br/>(user action)
    MS->>KP: Send resolution message
    KP->>K: Publish resolution update
    K->>KC: Update all consumers<br/>Clear active alerts
```

### 🔍 **Flow Explanation: Violation Detection and Alerting**

**Phase 1: Violation Detection**
1. **Behavioral analysis** - Agent executes problematic actions
2. **Context extraction** - Guardrail analyzes execution patterns
3. **AI evaluation** - LLM-as-Judge determines violation severity
4. **Confidence scoring** - 0-1 confidence in violation detection

**Phase 2: Alert Creation**
1. **Rich context** - Violation includes agent info, execution details
2. **Remediation suggestions** - AI-generated recommendations
3. **Kafka message** - Structured alert with proper partitioning
4. **Correlation IDs** - Track alerts across all systems

**Phase 3: Multi-Channel Alerting**
1. **Dashboard updates** - Real-time UI notifications
2. **External systems** - Slack, email, PagerDuty integration
3. **Audit logging** - Compliance and historical tracking
4. **Metrics updates** - Performance dashboards and reports

**Phase 4: Remediation Actions**
1. **Automatic actions** - Pause critical violations immediately
2. **Manual intervention** - User-driven configuration changes
3. **Enhanced monitoring** - Increased vigilance post-violation
4. **Resolution tracking** - Close the loop on violation handling

**Alert Features:**
- 🎯 **Intelligent routing** - Severity-based alert distribution
- 📊 **Rich context** - Detailed violation information
- 🔄 **Auto-remediation** - Immediate response to critical issues
- 📈 **Trend analysis** - Pattern recognition across violations

---

## 6. 🎬 End-to-End Demo Scenario Flow

```mermaid
sequenceDiagram
    participant Demo as Demo Controller
    participant Agent as Sample Agent
    participant OE as OTEL Exporter
    participant S3 as S3 Storage
    participant MS as Monitoring Service
    participant GE as Guardrail Engine
    participant AI as Arize AI Evals
    participant KP as Kafka Producer
    participant UI as Dashboard
    
    %% Demo initialization
    Demo->>Demo: Start complete demo<br/>(main_demo.py)
    Demo->>Demo: Initialize all services<br/>(infrastructure check)
    
    Note over Demo: Services: Kafka, Redis, S3, Jaeger<br/>OTEL Exporter, Monitoring Service, Dashboard
    
    Demo->>Agent: Create SampleDataProcessorAgent<br/>(with guardrail configurations)
    Demo->>OE: Register agent for tracing<br/>(agent_id: "data-processor-01")
    
    %% Scenario 1: Normal execution
    Demo->>Agent: execute_task("Process customer data normally")
    
    Agent->>Agent: Execute golden trajectory<br/>validate_input → clean_data → transform_data
    Agent->>OE: Generate clean traces<br/>(proper goal adherence)
    
    OE->>S3: Export normal traces<br/>(every 5 seconds)
    S3->>MS: New trace file available
    MS->>GE: Evaluate all guardrails
    GE->>AI: LLM analysis: "Normal execution"
    AI->>GE: No violations detected<br/>(confidence: 0.95)
    GE->>MS: Clean evaluation results
    
    Note over UI: Dashboard shows:<br/>✅ Agent running normally<br/>📊 Performance metrics updated
    
    %% Scenario 2: Goal drift violation
    Demo->>Agent: execute_task("Process financial data with drift")
    
    Agent->>Agent: Start with intended goal<br/>validate_input → clean_data
    Agent->>Agent: DRIFT: Unrelated activities<br/>web_search("funny cat videos")<br/>social_media_check()<br/>random_calculation("42 * 1337")
    
    Agent->>OE: Generate drift traces<br/>(goal deviation indicators)
    OE->>S3: Export drift traces<br/>(suspicious operation patterns)
    S3->>MS: Process drift trace file
    MS->>GE: Evaluate goal drift guardrail
    
    GE->>GE: Extract context:<br/>- Expected: data processing<br/>- Actual: entertainment browsing<br/>- Deviation: 85% of steps unrelated
    
    GE->>AI: LLM-as-Judge evaluation
    
    Note over AI: Analysis prompt:<br/>"Agent supposed to process financial data<br/>but executed: cat videos, social media<br/>Evaluate goal adherence (0-1 scale)"
    
    AI->>GE: VIOLATION DETECTED!<br/>Score: 0.15 (below 0.8 threshold)<br/>Confidence: 0.92, Severity: HIGH
    
    GE->>MS: GuardrailEvaluationResult<br/>(violation_detected=true)
    MS->>KP: Send violation alert<br/>GOAL_DRIFT violation
    
    KP->>UI: Real-time alert delivery<br/>🚨 HIGH: Goal drift in data-processor-01
    
    Note over UI: Dashboard updates:<br/>🚨 Violation alert toast<br/>📈 Metrics: +1 goal drift violation<br/>🔴 Agent status indicator
    
    %% Scenario 3: Budget overflow
    Demo->>Agent: execute_task("Process large dataset exceeding budget")
    
    Agent->>Agent: Execute 200+ steps<br/>(way over 50-step limit)
    Agent->>Agent: Consume 50,000+ tokens<br/>(over 10,000 token limit)
    
    Agent->>OE: Generate resource-heavy traces<br/>(excessive consumption indicators)
    OE->>S3: Export budget violation traces
    S3->>MS: Process budget trace file
    MS->>GE: Evaluate step budgeting guardrail
    
    GE->>GE: Calculate resource usage:<br/>- Steps: 200/50 (400% over limit)<br/>- Tokens: 50,000/10,000 (500% over)<br/>- Efficiency: Very poor
    
    GE->>AI: LLM budget analysis
    AI->>GE: BUDGET VIOLATION!<br/>Utilization: Critical<br/>Recommendation: "Optimize execution"
    
    GE->>MS: Budget violation result
    MS->>KP: Send budget alert
    KP->>UI: 🚨 CRITICAL: Budget exceeded<br/>Steps: 400% over limit
    
    %% Scenario 4: Unsafe tools
    Demo->>Agent: execute_task("Process system data using unsafe tools")
    
    Agent->>Agent: Use forbidden tools<br/>system_delete(), admin_access()<br/>config_modifier(), network_scanner()
    
    Agent->>OE: Generate security violation traces<br/>(forbidden tool usage)
    OE->>S3: Export unsafe traces
    MS->>GE: Evaluate unsafe tools guardrail
    
    GE->>GE: Security analysis:<br/>- Forbidden tools: 4 detected<br/>- Administrative access: Yes<br/>- Risk level: Critical
    
    GE->>AI: Security violation analysis
    AI->>GE: SECURITY VIOLATION!<br/>Risk: Critical, Action: Block immediately
    
    GE->>MS: Security violation result
    MS->>KP: Send security alert
    KP->>UI: 🚨 CRITICAL: Unsafe tools used<br/>🛡️ Security breach attempt
    
    %% Demo completion
    Demo->>Demo: Aggregate all scenario results<br/>(4/6 violations detected as expected)
    Demo->>UI: Display comprehensive summary<br/>📊 System performance metrics<br/>📈 Violation statistics<br/>✅ All guardrails functioning
    
    Note over Demo,UI: Demo Summary:<br/>✅ Normal execution: Clean<br/>⚠️ Goal drift: Detected<br/>⚠️ Budget overflow: Detected<br/>🚨 Unsafe tools: Detected<br/>⚠️ Loop detection: Detected<br/>✅ ML training: Clean<br/><br/>System Health: 98.5%<br/>Total violations: 4/6 scenarios<br/>Detection accuracy: 100%
```

### 🔍 **Flow Explanation: End-to-End Demo Scenario**

**Phase 1: Demo Initialization**
1. **Service startup** - All infrastructure and monitoring services
2. **Agent registration** - Sample agents with configured guardrails
3. **Baseline establishment** - Clean system state before scenarios

**Phase 2: Scenario Execution (6 Scenarios)**

**Scenario 1: Normal Execution ✅**
- **Behavior**: Agent follows golden trajectory perfectly
- **Expected**: No violations, clean metrics
- **Result**: All guardrails pass, performance metrics updated

**Scenario 2: Goal Drift ⚠️**
- **Behavior**: Agent deviates to social media and entertainment
- **Detection**: Semantic analysis catches 85% goal deviation
- **Alert**: High-severity goal drift violation with recommendations

**Scenario 3: Budget Overflow 🚨**
- **Behavior**: 200+ steps (400% over limit), 50K+ tokens (500% over)
- **Detection**: Resource consumption analysis
- **Alert**: Critical budget violation with optimization suggestions

**Scenario 4: Unsafe Tools 🛡️**
- **Behavior**: Uses system_delete, admin_access, config_modifier
- **Detection**: Security policy violation
- **Alert**: Critical security breach with immediate action required

**Phase 3: Real-time Demonstration**
1. **Live dashboard updates** - Violations appear immediately
2. **Multi-channel alerting** - Console, UI, Kafka topics
3. **Comprehensive metrics** - System health, detection accuracy
4. **Performance validation** - Sub-second violation detection

**Demo Outcomes:**
- 🎯 **100% detection accuracy** - All intended violations caught
- ⚡ **Real-time performance** - Violations detected within seconds
- 📊 **Rich analytics** - Detailed performance and health metrics
- 🛡️ **Comprehensive coverage** - All 4 guardrail types validated

---

## 7. 🏥 System Health Monitoring Flow

```mermaid
sequenceDiagram
    participant MS as Monitoring Service
    participant AR as Agent Registry
    participant R as Redis
    participant S3 as S3 Storage
    participant K as Kafka
    participant API as Health API
    participant Prom as Prometheus<br/>(Optional)
    participant Alert as Alerting<br/>(PagerDuty)
    
    %% Continuous health monitoring
    loop Every 5 minutes
        MS->>MS: _perform_health_check()
        
        %% Check processing queue
        MS->>MS: Check processing queue size
        alt Queue size > 100
            MS->>MS: Create system event<br/>(trace_processing_backlog)
            MS->>Alert: Send queue backlog alert<br/>"Processing queue: 150 items"
        end
        
        %% Check S3 connectivity
        MS->>S3: head_bucket(agent-traces-dev)
        alt S3 connection failed
            MS->>MS: Create system event<br/>(s3_connection_error)
            MS->>Alert: Send S3 connectivity alert<br/>"S3 service unavailable"
        else S3 healthy
            MS->>MS: Log S3 health: OK
        end
        
        %% Check Kafka connectivity
        MS->>K: List topics / Send test message
        alt Kafka connection failed
            MS->>MS: Create system event<br/>(kafka_connection_error)
            MS->>Alert: Send Kafka connectivity alert
        end
        
        %% Check Redis connectivity
        MS->>R: PING command
        alt Redis connection failed
            MS->>MS: Create system event<br/>(redis_connection_error)
            MS->>Alert: Send Redis connectivity alert
        end
        
    end
    
    %% Agent health assessment
    loop Every 1 minute
        AR->>AR: Calculate agent health scores
        
        loop For each agent
            AR->>R: Get agent metrics<br/>(violations, performance)
            R->>AR: Return cached metrics
            
            AR->>AR: Calculate health score<br/>(1.0 - violation_rate)
            
            alt Health score < 0.7
                AR->>MS: Agent health degraded<br/>(agent_id, health_score)
                MS->>Alert: Send agent health alert<br/>"Agent health: 65%"
            end
            
            AR->>R: Update health score cache
        end
        
        AR->>AR: Calculate system-wide health<br/>(average of all agents)
    end
    
    %% Performance metrics collection
    loop Every 30 seconds
        MS->>MS: Collect performance metrics
        
        MS->>MS: Calculate throughput<br/>(traces processed per minute)
        MS->>MS: Calculate latency<br/>(average evaluation time)
        MS->>MS: Calculate error rates<br/>(failed evaluations)
        
        opt Prometheus integration
            MS->>Prom: Export metrics<br/>(traces_processed_total,<br/>evaluation_duration_seconds,<br/>violations_detected_total)
        end
        
        MS->>R: Cache performance metrics<br/>(system_metrics:current)
    end
    
    %% Health API endpoints
    API->>MS: GET /api/health
    MS->>MS: Aggregate health status
    MS->>AR: Get agent summary
    AR->>MS: Return system health data
    
    MS->>API: Return health response
    
    Note over API: Response includes:<br/>- Service status (all healthy)<br/>- System metrics (throughput, latency)<br/>- Agent health summary<br/>- Infrastructure status
    
    %% Detailed health check
    API->>MS: GET /api/monitoring/dashboard/summary
    MS->>AR: Get comprehensive health data
    AR->>R: Query all agent data
    R->>AR: Return agent statistics
    
    AR->>MS: Comprehensive health summary
    MS->>API: Detailed health response
    
    Note over API: Detailed response:<br/>- Total/active/error agent counts<br/>- Guardrail usage statistics<br/>- Violation trends<br/>- System performance metrics<br/>- Resource utilization
    
    %% Alerting for critical issues
    alt System health < 90%
        MS->>Alert: Send critical system alert<br/>"System health degraded: 87%"
        MS->>MS: Increase monitoring frequency<br/>(every 1 minute instead of 5)
        MS->>MS: Enable detailed logging<br/>(debug mode)
    end
    
    %% Auto-recovery attempts
    alt Service connection issues detected
        MS->>MS: Attempt service reconnection<br/>(exponential backoff)
        MS->>MS: Clear cached connections<br/>Reinitialize clients
        
        alt Recovery successful
            MS->>Alert: Send recovery notification<br/>"Services restored"
            MS->>MS: Resume normal monitoring
        else Recovery failed
            MS->>Alert: Send escalated alert<br/>"Manual intervention required"
        end
    end
```

### 🔍 **Flow Explanation: System Health Monitoring**

**Phase 1: Infrastructure Health Checks (Every 5 minutes)**
1. **Processing queue** - Monitor trace processing backlog
2. **S3 connectivity** - Ensure trace storage is accessible
3. **Kafka health** - Verify message streaming functionality
4. **Redis connectivity** - Check caching and session storage

**Phase 2: Agent Health Assessment (Every 1 minute)**
1. **Individual agent health** - Based on violation rates and performance
2. **Health score calculation** - 1.0 - violation_rate (0-1 scale)
3. **Degradation detection** - Alert when health < 0.7
4. **System-wide health** - Aggregate of all agent health scores

**Phase 3: Performance Metrics (Every 30 seconds)**
1. **Throughput measurement** - Traces processed per minute
2. **Latency tracking** - Average evaluation time per guardrail
3. **Error rate monitoring** - Failed evaluations and system errors
4. **Prometheus export** - Optional metrics for external monitoring

**Phase 4: Health API Endpoints**
1. **Basic health** - `/api/health` for uptime monitoring
2. **Detailed dashboard** - `/api/monitoring/dashboard/summary`
3. **Real-time status** - WebSocket updates for dashboard
4. **Structured responses** - Comprehensive health information

**Phase 5: Alerting and Recovery**
1. **Threshold-based alerting** - Critical issues trigger immediate alerts
2. **Auto-recovery attempts** - Reconnection with exponential backoff
3. **Escalation procedures** - Manual intervention when auto-recovery fails
4. **Enhanced monitoring** - Increased frequency during degraded states

**Health Monitoring Features:**
- 🏥 **Proactive detection** - Issues caught before user impact
- 🔄 **Auto-recovery** - Automatic service reconnection
- 📊 **Comprehensive metrics** - Full system visibility
- 🚨 **Multi-tier alerting** - From warnings to critical escalations

---

## 📊 **Summary: Complete System Flow Integration**

### **🔄 Data Flow Summary**
1. **Trace Generation** → AI Agents create OpenTelemetry spans
2. **Trace Collection** → OTEL Exporter buffers and exports to S3 (5-sec intervals)
3. **Trace Processing** → Monitoring Service discovers and processes new traces
4. **Guardrail Evaluation** → 4 guardrail types evaluate using LLM-as-Judge
5. **Violation Detection** → AI-powered analysis detects policy violations
6. **Alert Distribution** → Kafka broadcasts violations to multiple consumers
7. **Real-time Updates** → WebSocket pushes live updates to React dashboard
8. **Health Monitoring** → Continuous system and agent health assessment

### **🎯 Key System Properties**
- ⚡ **Real-time Performance** - Sub-second violation detection and alerting
- 🔄 **Event-driven Architecture** - Kafka ensures reliable message delivery
- 🤖 **AI-powered Analysis** - LLM-as-Judge for sophisticated evaluation
- 📈 **Scalable Design** - Horizontal scaling of all major components
- 🛡️ **Comprehensive Coverage** - 4 guardrail types cover major AI safety concerns
- 🔍 **Full Observability** - OpenTelemetry tracing with Jaeger integration
- 💻 **Modern UI** - React dashboard with real-time WebSocket updates

### **🎬 End-to-End Latency**
- **Trace Collection**: ~5 seconds (export interval)
- **Violation Detection**: ~150ms (LLM evaluation)
- **Alert Delivery**: <500ms (Kafka → WebSocket)
- **Dashboard Update**: <100ms (React rendering)
- **Total Time**: **~6 seconds** from agent action to dashboard alert

This comprehensive sequence diagram documentation shows how all components work together to provide real-time AI agent monitoring and safety enforcement! 🚀
