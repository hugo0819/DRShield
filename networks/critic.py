import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Critic network, approximates the function Q(s, a).
    - Defines the network structure only, without training logic or target network.
    """
    def __init__(self, state_size, action_size, hidden_units=(256, 128), device=None):
        super(Critic, self).__init__()

        # State path
        self.state_hidden_1 = nn.Linear(state_size, hidden_units[0])
        
        # State + action path
        self.combined_hidden_2 = nn.Linear(hidden_units[0] + action_size, hidden_units[1])
        self.output = nn.Linear(hidden_units[1], 1)

    def forward(self, state, action):
        """Forward propagation."""
        state_out = F.relu(self.state_hidden_1(state))
        # Ensure concatenation along the correct dimension
        if state_out.dim() > 1 and action.dim() > 1:
             combined = torch.cat([state_out, action], dim=1)
        else:
             combined = torch.cat([state_out, action], dim=0)
        x = F.relu(self.combined_hidden_2(combined))
        q_value = self.output(x)
        return q_value