# GNN-PPO: Graph Neural Network-Enhanced Deep Reinforcement Learning for DDoS Mitigation in Software-Defined Networks

> **Anonymous Submission for Peer Review**

## Abstract

This repository provides the implementation of a novel framework for DDoS attack mitigation in Software-Defined Networks (SDN) using GPO. The proposed approach leverages the inherent graph structure of network topologies to learn optimal ML model deployment strategies, achieving effective attack detection while minimizing resource consumption.

## Key Contributions

- **Topology-Aware Decision Making**: A GNN-based encoder that captures the structural relationships between switches, enabling the agent to learn deployment policies that generalize across different network topologies.

- **Multi-Objective Reward Function**: A normalized reward formulation considering four critical metrics:
  - Mitigation Rate (MR): $f_{MR}(x) = x$
  - False Positive Rate (FPR): $f_{FPR}(x) = e^{-\alpha x}$
  - Resource Utilization (RU): $f_{RU}(x) = 1 - x$
  - Latency (LTD): $f_{LTD}(x) = e^{-x/\tau}$

- **Adaptive Model Placement**: Dynamic deployment of detection models at optimal network locations based on real-time traffic patterns and resource constraints.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SDN Controller                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Topology   │───▶│  Graph State │───▶│   GNN Encoder    │   │
│  │   Discovery  │    │    Builder   │    │   (GCN/GAT)      │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                                           ┌────────▼────────┐   │
│                                           │ Node Embeddings │   │
│                                           └────────┬────────┘   │
│                            ┌───────────────────────┼────────────┤
│                            ▼                       ▼            │
│                     ┌────────────┐          ┌───────────┐       │
│                     │   Actor    │          │  Critic   │       │
│                     │ (per-node) │          │ (global)  │       │
│                     └─────┬──────┘          └─────┬─────┘       │
│                           │                       │             │
│                     ┌─────▼──────┐          ┌─────▼─────┐       │
│                     │  Actions   │          │   Value   │       │
│                     │ Deploy/Not │          │ Estimate  │       │
│                     └────────────┘          └───────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
├── controller.py              # SDN controller with RL training loop
├── gnn_ppo_agent.py           # GPO agent implementation
├── ppo_agent.py               # Baseline PPO agent (MLP-based)
├── sdn_environment.py         # RL environment with graph state support
├── config.py                  # Hyperparameters and configurations
├── networks/
│   ├── gnn.py                 # GNN encoder (GCN/GAT)
│   ├── actor.py               # Actor network
│   └── critic.py              # Critic network
├── managers/
│   ├── graph_state_space.py   # Graph-based state representation
│   ├── reward_function.py     # Normalized reward calculator
│   └── abstract_model_manager.py  # Model deployment manager
├── tree_topology.py           # Network topology (Mininet)
└── saved_models/              # Trained model checkpoints
```

## Requirements

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mininet openvswitch-switch openvswitch-common
```

### Python Dependencies

```bash
# Create virtual environment (Python 3.8+)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch Geometric for optimized GNN
pip install torch-geometric
```

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_GNN_PPO` | `True` | Enable GNN-based agent |
| `GNN_TYPE` | `'gcn'` | GNN architecture (`'gcn'` or `'gat'`) |
| `GNN_HIDDEN_DIM` | `128` | Hidden dimension of GNN layers |
| `GNN_NUM_LAYERS` | `3` | Number of GNN layers |
| `PPO_LR` | `3e-4` | Learning rate |
| `PPO_GAMMA` | `0.99` | Discount factor |
| `PPO_EPS_CLIP` | `0.2` | PPO clipping parameter |

### Reward Function Weights

| Weight | Default | Metric |
|--------|---------|--------|
| `W_MR` | `0.4` | Mitigation Rate |
| `W_FPR` | `0.2` | False Positive Rate |
| `W_RU` | `0.2` | Resource Utilization |
| `W_LTD` | `0.2` | Latency |

## Running Experiments

### Training

```bash
# Terminal 1: Start the SDN controller with RL agent
ryu-manager controller.py

# Terminal 2: Start the network topology
sudo python3 tree_topology.py
```

### Evaluation

Trained models are saved in `saved_models/`. To evaluate:

```python
from gnn_ppo_agent import GNNPPOAgent
import config

agent = GNNPPOAgent(node_feature_dim=config.NODE_FEATURE_DIM)
agent.load_models("saved_models/gnn_ppo_episode_2000.pth")
```

### Switching Between Algorithms

To compare with baseline PPO (without GNN):

```python
# In config.py
USE_GNN_PPO = False  # Uses MLP-based PPO
```

## Node Features

Each switch node is represented by a 7-dimensional feature vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `cpu_available` | Available CPU ratio [0,1] |
| 1 | `memory_available` | Available memory ratio [0,1] |
| 2 | `has_ml_model` | ML model deployed {0,1} |
| 3 | `traffic_load` | Normalized traffic load [0,1] |
| 4 | `attack_detected` | Attack detection flag {0,1} |
| 5 | `degree` | Normalized node degree [0,1] |
| 6 | `is_edge_switch` | Edge switch indicator {0,1} |

## Experimental Setup

- **Network Simulator**: Mininet 2.3.0
- **SDN Controller**: Ryu 4.34
- **Switch**: Open vSwitch (OpenFlow 1.3)
- **Deep Learning**: PyTorch 2.0+
- **GNN Library**: PyTorch Geometric (optional)

## License

This project is released under the MIT License for academic and research purposes.

---

*This repository is provided for anonymous peer review. Author information will be added upon acceptance.*
