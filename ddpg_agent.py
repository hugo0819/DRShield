import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

# Import from project files
from networks.actor import Actor
from networks.critic import Critic
from networks.replay_buffer import ReplayBuffer
import config

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create neural networks
        self.actor = Actor(state_dim, action_dim).to(config.DEVICE)
        self.critic = Critic(state_dim, action_dim).to(config.DEVICE)
        
        # Target networks
        self.target_actor = Actor(state_dim, action_dim).to(config.DEVICE)
        self.target_critic = Critic(state_dim, action_dim).to(config.DEVICE)
        
        # Copy weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC)
        
        # Replay buffer
        self.memory = ReplayBuffer(config.REPLAY_MEM_CAPACITY)

    def get_action(self, state):
        """Select action based on state."""
        # Convert state to tensor
        state = torch.FloatTensor(state).to(config.DEVICE)
        
        # Use actor network to generate action
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        
        # Return action (no noise added, as noise is added in the controller)
        return action

    def update(self):
        """Update network parameters."""
        if len(self.memory) < config.MINI_BATCH_SIZE:
            return
            
        # Sample from replay buffer
        batch = self.memory.sample_batch(config.MINI_BATCH_SIZE)
        state, action, reward, next_state, done = batch
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(config.DEVICE)
        action = torch.FloatTensor(action).to(config.DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(config.DEVICE)
        next_state = torch.FloatTensor(next_state).to(config.DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(config.DEVICE)
        
        # Update Critic
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q = reward + (1 - done) * config.GAMMA * target_q
            
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.TAU) + param.data * config.TAU
            )
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - config.TAU) + param.data * config.TAU
            )

    def save_models(self, episode):
        """Save model weights."""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)
        
        torch.save(self.actor.state_dict(), f"{config.MODEL_SAVE_PATH}/actor_episode_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{config.MODEL_SAVE_PATH}/critic_episode_{episode}.pth")
        print(f"Models saved at episode {episode}.")