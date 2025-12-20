"""
SDN DDoS Defense System - Neural Network Module

Includes:
- Actor: Actor network for DDPG/PPO
- Critic: Critic network for DDPG/PPO
- ReplayBuffer: Experience replay buffer
- GNN Module: Graph Neural Network components
"""

from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer

# GNN Module
try:
    from .gnn import (
        GNNEncoder,
        GNNActorCritic,
        GraphData,
        GraphBatch,
        create_graph_data,
        TORCH_GEOMETRIC_AVAILABLE
    )
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

__all__ = [
    'Actor',
    'Critic', 
    'ReplayBuffer',
    'GNNEncoder',
    'GNNActorCritic',
    'GraphData',
    'GraphBatch',
    'create_graph_data',
    'TORCH_GEOMETRIC_AVAILABLE'
]

