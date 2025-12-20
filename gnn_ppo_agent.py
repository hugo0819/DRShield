"""
GNN+PPO Agent
Combines Graph Neural Networks and Proximal Policy Optimization
for DDoS defense decisions in SDN networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import os

import config
from networks.gnn import GNNActorCritic, GraphData, GraphBatch


class GNNPPOMemory:
    """
    GNN+PPO trajectory storage
    Stores graph data and corresponding actions, rewards, etc.
    """
    
    def __init__(self):
        self.graph_data_list = []  # Stores graph data
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, graph_data, action, log_prob, reward, value, done):
        """Store data for one timestep."""
        self.graph_data_list.append(graph_data)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear stored data."""
        self.graph_data_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.rewards)
    
    def get_batch(self, device):
        """
        Retrieve batch data.

        Returns:
            batched_graph: Batched graph data
            actions: Action tensor
            log_probs: Log probability tensor
            rewards: Reward list
            values: Value tensor
            dones: Done flags list
        """
        # Batch process graph data
        batched_graph = GraphBatch.from_data_list(self.graph_data_list)
        if hasattr(batched_graph, 'to'):
            batched_graph = batched_graph.to(device)
        
        # Convert to tensors
        actions = torch.stack(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        values = torch.stack(self.values).to(device)
        
        return batched_graph, actions, log_probs, self.rewards, values, self.dones


class GNNPPOAgent:
    """
    GNN+PPO Agent

    Features:
    - Uses GNN encoder to capture network topology
    - Uses PPO algorithm for policy optimization
    - Supports discrete actions (switch deployment decisions)
    """
    
    def __init__(self, node_feature_dim, 
                 hidden_dim=128,
                 num_gnn_layers=3,
                 gnn_type='gcn',
                 lr=3e-4, 
                 gamma=0.99, 
                 eps_clip=0.2,
                 k_epochs=10, 
                 gae_lambda=0.95,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5):
        """
        Initialize GNN+PPO Agent.

        Args:
            node_feature_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN type ('gcn' or 'gat')
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of iterations per update
            gae_lambda: GAE λ parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Gradient clipping norm
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = config.DEVICE
        
        # GNN+Actor-Critic network
        self.policy = GNNActorCritic(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=0.1
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        
        # Experience storage
        self.memory = GNNPPOMemory()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def get_action(self, graph_data):
        """
        Select action based on graph state.

        Args:
            graph_data: Graph data object

        Returns:
            action: Action array (deployment decisions for each switch)
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Ensure data is on the correct device
        if hasattr(graph_data, 'to'):
            graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(graph_data)
        
        return action.cpu().numpy().astype(int), log_prob, value
    
    def store_transition(self, graph_data, action, log_prob, reward, value, done):
        """Store experience."""
        # Clone graph data to avoid reference issues
        cloned_data = GraphData(
            x=graph_data.x.clone(),
            edge_index=graph_data.edge_index.clone()
        )
        
        action_tensor = torch.FloatTensor(action).to(self.device)
        self.memory.store(cloned_data, action_tensor, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: Value estimation tensor
            dones: List of done flags
            next_value: Value estimate for the next state

        Returns:
            advantages: Advantage estimates
            returns: Returns
        """
        advantages = []
        gae = 0
        
        values_list = values.cpu().tolist()
        values_list.append(next_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_list[t+1] * (1 - dones[t]) - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_value=0):
        """
        PPO policy update.

        Args:
            next_value: Value estimate for the next state (used for GAE computation)
        """
        if len(self.memory) == 0:
            return
        
        # Get batch data
        batched_graph, actions, old_log_probs, rewards, values, dones = \
            self.memory.get_batch(self.device)
        
        # Calculate GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Record losses
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_entropy = 0
        
        # K updates
        for _ in range(self.k_epochs):
            # Since GNN needs to handle graph structure, we need to process one by one or use batched graph
            # Here we use a simplified way: process one by one
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            for i, graph_data in enumerate(self.memory.graph_data_list):
                graph_data = graph_data.to(self.device)
                
                # Forward pass
                log_prob, state_value, entropy = self.policy.evaluate(
                    graph_data, actions[i]
                )
                
                # Importance sampling ratio
                ratio = torch.exp(log_prob - old_log_probs[i])
                
                # PPO-Clip objective
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                
                # Loss calculation
                policy_loss = -torch.min(surr1, surr2)
                value_loss = nn.MSELoss()(state_value, returns[i])
                entropy_loss = -entropy
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
            
            # Average loss
            batch_size = len(self.memory)
            avg_policy_loss = total_policy_loss / batch_size
            avg_value_loss = total_value_loss / batch_size
            avg_entropy = total_entropy / batch_size
            
            # Total loss
            loss = (avg_policy_loss + 
                   self.value_loss_coef * avg_value_loss + 
                   self.entropy_coef * (-avg_entropy))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            epoch_policy_loss += avg_policy_loss.item()
            epoch_value_loss += avg_value_loss.item()
            epoch_entropy += avg_entropy.item()
        
        # Learning rate scheduling
        self.scheduler.step()
        
        # Record statistics
        self.training_stats['policy_loss'].append(epoch_policy_loss / self.k_epochs)
        self.training_stats['value_loss'].append(epoch_value_loss / self.k_epochs)
        self.training_stats['entropy'].append(epoch_entropy / self.k_epochs)
        
        # Clear experience
        self.memory.clear()
    
    def save_models(self, episode):
        """Save models."""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)
        
        save_path = f"{config.MODEL_SAVE_PATH}/gnn_ppo_episode_{episode}.pth"
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }, save_path)
        print(f"GNN+PPO model saved at episode {episode} to {save_path}")
    
    def load_models(self, path):
        """Load models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        print(f"GNN+PPO model loaded successfully from {path}")
    
    def get_training_stats(self):
        """Retrieve training statistics."""
        return self.training_stats


def test_gnn_ppo_agent():
    """Test GNN+PPO Agent"""
    print("Testing GNN+PPO Agent...")
    
    # Create test graph data
    num_nodes = 7
    node_feature_dim = 7
    
    # Random node features
    node_features = np.random.rand(num_nodes, node_feature_dim).astype(np.float32)
    
    # Tree-like edges
    edge_list = [
        (0, 1), (1, 0),  # s1 - s2
        (0, 2), (2, 0),  # s1 - s3
        (1, 3), (3, 1),  # s2 - s4
        (1, 4), (4, 1),  # s2 - s5
        (2, 5), (5, 2),  # s3 - s6
        (2, 6), (6, 2),  # s3 - s7
    ]
    
    # Create graph data
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    graph_data = GraphData(x=x, edge_index=edge_index)
    
    # Create agent
    agent = GNNPPOAgent(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        num_gnn_layers=2,
        gnn_type='gcn'
    )
    
    # Test action selection
    action, log_prob, value = agent.get_action(graph_data)
    print(f"Action: {action}")
    print(f"Log Probability: {log_prob}")
    print(f"State Value: {value}")
    
    # Test storage and update
    for _ in range(10):
        action, log_prob, value = agent.get_action(graph_data)
        reward = np.random.rand()
        done = False
        agent.store_transition(graph_data, action, log_prob, reward, value, done)
    
    agent.update()
    print("Update complete!")
    print(f"Training statistics: {agent.get_training_stats()}")
    
    print("GNN+PPO Agent test passed!")


if __name__ == "__main__":
    test_gnn_ppo_agent()

