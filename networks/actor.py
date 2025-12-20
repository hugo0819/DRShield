import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network, approximates the function u(s) -> a.
    - Defines the network structure only, without training logic or target network.
    """
    def __init__(self, state_size, action_size, hidden_units=(256, 128), device=None):
        super(Actor, self).__init__()

        # Define network structure
        self.hidden_1 = nn.Linear(state_size, hidden_units[0])
        self.hidden_2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.output_layer = nn.Linear(hidden_units[1], action_size)

    def forward(self, state):
        """Forward propagation."""
        x = F.relu(self.hidden_1(state))
        x = F.relu(self.hidden_2(x))
        actions = torch.sigmoid(self.output_layer(x)) 
        return actions