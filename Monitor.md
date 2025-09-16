# Kill Signal System Design for Rogue AI Agents in OCP/GKE

This document outlines the design for broadcasting a kill signal to AI agent applications running across multiple pods in Kubernetes-based environments (OCP or GKE). The system ensures near-simultaneous delivery without significant delays. Two options are presented: 

1. **Kubernetes-Native Scaling to Zero** (preferred for simplicity and low latency).
2. **Pub/Sub with NATS** (for custom signaling without disrupting deployment scaling).

Each option includes:
- Architecture components.
- High-level flow diagram (text-based).
- Sequence diagram (using Mermaid syntax for rendering in Markdown viewers like GitHub or VS Code).

## Option 1: Kubernetes-Native Scaling to Zero

This approach uses Kubernetes' built-in Deployment scaling to broadcast the kill signal implicitly via pod termination. A single API call triggers parallel SIGTERM signals to all pods.

### Architecture Components
- **Monitoring App**: Detects rogue AI and patches the Deployment.
- **Kubernetes API Server**: Handles the patch request and propagates changes.
- **Deployment Controller**: Reconciles the spec and initiates pod deletions.
- **Kubelets**: Node agents that send SIGTERM/SIGKILL to pods.
- **AI Agent Pods**: Managed by a Deployment; receive signals for termination.

### High-Level Flow Diagram (Text-Based)
```
[Monitoring App] --> Detect Rogue --> Patch Deployment (replicas=0) --> [API Server]
                                           |
                                           v
[API Server] --> Update Spec --> [Deployment Controller] --> Reconcile & Delete Pods
                                           |
                                           v (Parallel Broadcast)
[Kubelet Node 1] <-- Watch --> Terminate Pod A (SIGTERM --> SIGKILL if timeout)
[Kubelet Node 2] <-- Watch --> Terminate Pod B (SIGTERM --> SIGKILL if timeout)
... (For all pods)
                                           |
                                           v
Pods Terminated --> Optional: Collect Logs/Metrics
```

### Sequence Diagram
```mermaid
sequenceDiagram
    participant Monitor as Monitoring App
    participant API as Kubernetes API Server
    participant Controller as Deployment Controller
    participant Kubelet1 as Kubelet Node 1
    participant Kubelet2 as Kubelet Node 2
    participant Pod1 as AI Pod 1
    participant Pod2 as AI Pod 2

    Monitor->>API: Patch Deployment Scale (replicas=0)
    API->>Controller: Update Deployment Spec
    Controller->>API: Reconcile - Mark Pods for Deletion
    API->>Kubelet1: Watch Event - Delete Pod1
    API->>Kubelet2: Watch Event - Delete Pod2
    Note over Kubelet1,Kubelet2: Parallel Delivery
    Kubelet1->>Pod1: Send SIGTERM
    Kubelet2->>Pod2: Send SIGTERM
    Pod1->>Pod1: Graceful Shutdown (Handler)
    Pod2->>Pod2: Graceful Shutdown (Handler)
    alt Timeout (Grace Period)
        Kubelet1->>Pod1: Send SIGKILL
        Kubelet2->>Pod2: Send SIGKILL
    end
    Kubelet1->>API: Pod1 Terminated
    Kubelet2->>API: Pod2 Terminated
    API->>Monitor: Optional - Status Check
```

## Option 2: Pub/Sub with NATS

This approach deploys a NATS broker for pub/sub messaging. The monitoring app publishes a kill message, which is fanned out to subscribers in each pod for self-termination.

### Architecture Components
- **Monitoring App**: Detects rogue AI and publishes to NATS.
- **NATS Broker**: Lightweight pub/sub server (deployed in-cluster).
- **AI Agent Pods**: Each runs a NATS subscriber; self-sends SIGTERM on message receipt.
- **Kubernetes Network**: Facilitates in-cluster communication via Services.

### High-Level Flow Diagram (Text-Based)
```
[Monitoring App] --> Detect Rogue --> Publish 'kill' Message --> [NATS Broker]
                                           |
                                           v (Fan-Out)
[NATS Broker] --> Deliver Message --> [Pod1 Listener] --> Self SIGTERM --> Shutdown & Exit
                  --> Deliver Message --> [Pod2 Listener] --> Self SIGTERM --> Shutdown & Exit
... (Parallel for all pods)
                                           |
                                           v
Optional: Monitor Confirms Terminations --> Fallback Delete if Needed
```

### Sequence Diagram
```mermaid
sequenceDiagram
    participant Monitor as Monitoring App
    participant NATS as NATS Broker
    participant Pod1 as AI Pod 1 (Listener)
    participant Pod2 as AI Pod 2 (Listener)

    Note over Pod1,Pod2: Pods Subscribe on Startup
    Pod1->>NATS: Subscribe to "kill.channel"
    Pod2->>NATS: Subscribe to "kill.channel"
    Monitor->>NATS: Publish 'kill' on "kill.channel"
    NATS->>Pod1: Deliver Message
    NATS->>Pod2: Deliver Message
    Note over Pod1,Pod2: Parallel Receipt
    Pod1->>Pod1: Send SIGTERM to Self
    Pod2->>Pod2: Send SIGTERM to Self
    Pod1->>Pod1: Graceful Shutdown (Handler)
    Pod2->>Pod2: Graceful Shutdown (Handler)
    alt No Exit (Rare)
        Monitor->>Pod1: Fallback - Kubernetes Delete
    end
    Pod1->>NATS: Connection Closes (Implicit)
    Pod2->>NATS: Connection Closes (Implicit)
    Monitor->>Monitor: Optional - Poll Pod Status
```
