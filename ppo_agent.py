import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import config

class PPOActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(PPOActorCritic, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head: Outputs the probability of deployment for each switch
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Sigmoid()  # Outputs probabilities in [0,1]
        )
        
        # Critic head: Outputs state value
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def act(self, state):
        """Select action and return log probability"""
        action_probs, state_value = self.forward(state)
        
        # Sample deployment decisions for each switch using Bernoulli distribution
        dist = Bernoulli(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, state_value
    
    def evaluate(self, state, action):
        """Evaluate the log probability and entropy of actions"""
        action_probs, state_value = self.forward(state)
        
        dist = Bernoulli(action_probs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, state_value.squeeze(), entropy


class PPOMemory:
    """Trajectory storage for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def get_batch(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.values).squeeze(),
            torch.tensor(self.dones, dtype=torch.float32)
        )


class PPOAgent:
    """PPO Agent."""
    
    def __init__(self, state_dim, action_dim, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=10, gae_lambda=0.95):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        
        self.policy = PPOActorCritic(state_dim, action_dim).to(config.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = PPOMemory()
        self.device = config.DEVICE
    
    def get_action(self, state):
        """Select action"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)
        
        return action.cpu().numpy().astype(int), log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store experience"""
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        self.memory.store(state, action, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        values = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        
        return advantages, returns
    
    def update(self, next_value=0):
        """PPO update"""
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batch()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards.tolist(), values, dones.tolist(), next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # K epochs of updates
        for _ in range(self.k_epochs):
            log_probs, state_values, entropy = self.policy.evaluate(states, actions)
            
            # Importance sampling ratio
            ratios = torch.exp(log_probs - old_log_probs)
            
            # PPO-Clip objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Loss function
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.memory.clear()
    
    def save_models(self, episode):
        """Save models"""
        import os
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)
        
        torch.save(self.policy.state_dict(), 
                   f"{config.MODEL_SAVE_PATH}/ppo_policy_episode_{episode}.pth")
        print(f"PPO model saved at episode {episode}.")
    
    def load_models(self, path):
        """Load models"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))