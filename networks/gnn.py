"""
GNN (Graph Neural Network) module
Used to capture SDN network topology information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torch_geometric, use custom implementation if not available
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. Using custom GNN implementation.")


class CustomGCNConv(nn.Module):
    """Custom GCN convolution layer (used when torch_geometric is unavailable)."""
    
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Node features [num_nodes, in_channels]
        edge_index: Edge indices [2, num_edges]
        """
        num_nodes = x.size(0)
        
        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=x.device)
        
        # Degree normalization (D^-0.5 * A * D^-0.5)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        
        # Message passing
        x = torch.matmul(norm_adj, x)
        x = self.linear(x)
        
        return x


class CustomGATConv(nn.Module):
    """Custom GAT convolution layer (used when torch_geometric is unavailable)."""
    
    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.1):
        super(CustomGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, in_channels]
        edge_index: Edge indices [2, num_edges]
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x).view(num_nodes, self.heads, self.out_channels)
        
        # Build adjacency matrix (with self-loops)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + torch.eye(num_nodes, device=x.device)
        
        # Compute attention scores
        outputs = []
        for head in range(self.heads):
            h_head = h[:, head, :]  # [num_nodes, out_channels]
            
            # Compute attention coefficients
            a_input = torch.cat([
                h_head.unsqueeze(1).repeat(1, num_nodes, 1),
                h_head.unsqueeze(0).repeat(num_nodes, 1, 1)
            ], dim=-1)  # [num_nodes, num_nodes, 2*out_channels]
            
            e = self.leaky_relu(torch.matmul(a_input, self.a[head]))  # [num_nodes, num_nodes]
            
            # Mask non-neighboring nodes
            mask = (adj == 0)
            e = e.masked_fill(mask, float('-inf'))
            
            # Softmax normalization
            alpha = F.softmax(e, dim=1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # Weighted aggregation
            out = torch.matmul(alpha, h_head)
            outputs.append(out)
        
        if self.concat:
            return torch.cat(outputs, dim=-1)
        else:
            return torch.stack(outputs, dim=0).mean(dim=0)


def custom_global_mean_pool(x, batch=None):
    """Global average pooling"""
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    else:
        # Group by batch and compute mean
        unique_batches = torch.unique(batch)
        outputs = []
        for b in unique_batches:
            mask = (batch == b)
            outputs.append(x[mask].mean(dim=0))
        return torch.stack(outputs, dim=0)


# Select the GNN layer implementation to use
if TORCH_GEOMETRIC_AVAILABLE:
    GCNLayer = GCNConv
    GATLayer = GATConv
    global_pool = global_mean_pool
    GraphData = Data
    GraphBatch = Batch
else:
    GCNLayer = CustomGCNConv
    GATLayer = CustomGATConv
    global_pool = custom_global_mean_pool
    
    class GraphData:
        """Simplified graph data structure"""
        def __init__(self, x, edge_index, edge_attr=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
        
        def to(self, device):
            self.x = self.x.to(device)
            self.edge_index = self.edge_index.to(device)
            if self.edge_attr is not None:
                self.edge_attr = self.edge_attr.to(device)
            return self
    
    class GraphBatch:
        """Simplified batch graph data structure"""
        @staticmethod
        def from_data_list(data_list):
            if len(data_list) == 1:
                return data_list[0]
            
            xs = []
            edge_indices = []
            batch = []
            node_offset = 0
            
            for i, data in enumerate(data_list):
                xs.append(data.x)
                edge_indices.append(data.edge_index + node_offset)
                batch.extend([i] * data.x.size(0))
                node_offset += data.x.size(0)
            
            combined = GraphData(
                x=torch.cat(xs, dim=0),
                edge_index=torch.cat(edge_indices, dim=1)
            )
            combined.batch = torch.tensor(batch)
            return combined


class GNNEncoder(nn.Module):
    """
    GNN Encoder - Extracts features from network topology.
    Supports GCN and GAT architectures.
    """
    
    def __init__(self, node_feature_dim, hidden_dim=128, num_layers=3, 
                 gnn_type='gcn', dropout=0.1, heads=4):
        super(GNNEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNLayer(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                if TORCH_GEOMETRIC_AVAILABLE:
                    self.convs.append(GATLayer(hidden_dim, hidden_dim // heads, 
                                               heads=heads, concat=True, dropout=dropout))
                else:
                    self.convs.append(GATLayer(hidden_dim, hidden_dim // heads,
                                               heads=heads, concat=True, dropout=dropout))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data):
        """
        Forward propagation.
        Args:
            data: Graph data object, containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge indices [2, num_edges]
        Returns:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            graph_embedding: Graph-level embedding [1, hidden_dim]
        """
        x = data.x
        edge_index = data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN message passing
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection
        
        # Output projection
        node_embeddings = self.output_proj(x)
        
        # Global pooling to get graph-level embedding
        batch = getattr(data, 'batch', None)
        graph_embedding = global_pool(node_embeddings, batch)
        
        return node_embeddings, graph_embedding


class GNNActorCritic(nn.Module):
    """
    GNN-based Actor-Critic network
    - Uses GNN encoder to capture network topology
    - Actor: Outputs deployment probabilities for each switch node
    - Critic: Estimates global state value
    """
    
    def __init__(self, node_feature_dim, hidden_dim=128, num_gnn_layers=3,
                 gnn_type='gcn', dropout=0.1):
        super(GNNActorCritic, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GNN encoder
        self.gnn_encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Actor head: Outputs a deployment probability for each node
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Critic head: Estimates state value from graph-level embedding
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        """
        Forward propagation.
        Args:
            data: Graph data object
        Returns:
            action_probs: Action probabilities for each node [num_nodes]
            state_value: State value [1]
        """
        # GNN encoding
        node_embeddings, graph_embedding = self.gnn_encoder(data)
        
        # Actor: Deployment probability for each node
        action_probs = self.actor(node_embeddings).squeeze(-1)
        
        # Critic: Global state value
        state_value = self.critic(graph_embedding)
        
        return action_probs, state_value
    
    def act(self, data):
        """Select actions and return log probabilities."""
        from torch.distributions import Bernoulli
        
        action_probs, state_value = self.forward(data)
        
        # Sample actions for each node using Bernoulli distribution
        dist = Bernoulli(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        return action, log_prob, state_value.squeeze()
    
    def evaluate(self, data, action):
        """Evaluate log probabilities and entropy of actions."""
        from torch.distributions import Bernoulli
        
        action_probs, state_value = self.forward(data)
        
        dist = Bernoulli(action_probs)
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().sum()
        
        return log_prob, state_value.squeeze(), entropy


def create_graph_data(node_features, edge_list, device='cpu'):
    """
    Helper function to create graph data object.

    Args:
        node_features: Node feature array [num_nodes, feature_dim]
        edge_list: Edge list [(src, dst), ...]
        device: Computing device

    Returns:
        GraphData object
    """
    x = torch.FloatTensor(node_features).to(device)
    
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t().contiguous().to(device)
    else:
        edge_index = torch.LongTensor([[], []]).to(device)
    
    return GraphData(x=x, edge_index=edge_index)

