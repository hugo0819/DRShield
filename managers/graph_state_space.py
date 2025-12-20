"""
Graph State Space Manager
Converts SDN network topology into a graph structure for GNN processing.
"""

import numpy as np
import torch
from ryu.topology import api as topo_api


class GraphStateSpace:
    """
    Graph State Space Manager
    - Constructs graph structure from SDN topology
    - Extracts node features (switch states)
    - Extracts edge features (link states)
    """
    
    def __init__(self, model_manager, controller, device='cpu'):
        """
        Initialize the graph state space.

        Args:
            model_manager: Model manager (provides deployment states)
            controller: SDN controller (provides topology information)
            device: Computing device
        """
        self.model_manager = model_manager
        self.controller = controller
        self.device = device
        
        # Node feature dimensions
        # [cpu_available, memory_available, has_ml_model, traffic_load, 
        #  attack_detected, degree, is_edge_switch]

        # Edge feature dimensions
        # [bandwidth_utilization, latency, is_active]

        # Cached topology information
        self._cached_links = None
        self._cached_switches = None
    
    def get_graph_state(self):
        """
        Get the current graph state of the network.

        Returns:
            node_features: Node feature array [num_nodes, node_feature_dim]
            edge_list: Edge list [(src_idx, dst_idx), ...]
            switch_id_to_idx: Mapping of switch IDs to indices
        """
        # 获取拓扑信息
        switches = self._get_switches()
        links = self._get_links()
        
        # 构建交换机ID到索引的映射
        switch_ids = sorted([sw_id for sw_id in switches])
        switch_id_to_idx = {sw_id: idx for idx, sw_id in enumerate(switch_ids)}
        idx_to_switch_id = {idx: sw_id for sw_id, idx in switch_id_to_idx.items()}
        
        num_nodes = len(switch_ids)
        
        # 提取节点特征
        node_features = np.zeros((num_nodes, self.node_feature_dim))
        for sw_id in switch_ids:
            idx = switch_id_to_idx[sw_id]
            node_features[idx] = self._get_node_features(sw_id, switches, links)
        
        # 构建边列表（双向边）
        edge_list = []
        for link in links:
            src_id = link['src_dpid']
            dst_id = link['dst_dpid']
            if src_id in switch_id_to_idx and dst_id in switch_id_to_idx:
                src_idx = switch_id_to_idx[src_id]
                dst_idx = switch_id_to_idx[dst_id]
                edge_list.append((src_idx, dst_idx))
                edge_list.append((dst_idx, src_idx))  # 双向
        
        return node_features, edge_list, switch_id_to_idx, idx_to_switch_id
    
    def _get_switches(self):
        """Retrieve all switch information."""
        switches = {}
        
        # 从控制器获取交换机状态
        for sw_id, state in self.model_manager.switch_states.items():
            switches[sw_id] = {
                'cpu_available': state.get('cpu_available', 1.0),
                'memory_available': state.get('memory_available', 1.0),
                'has_ml_model': state.get('has_ml_model', False),
                'traffic_load': state.get('traffic_load', 0.0),
                'attack_detected': state.get('attack_detected', False)
            }
        
        return switches
    
    def _get_links(self):
        """Retrieve all link information."""
        links = []
        
        try:
            # 尝试从Ryu拓扑API获取链路
            ryu_links = topo_api.get_all_link(self.controller)
            for link in ryu_links:
                links.append({
                    'src_dpid': link.src.dpid,
                    'dst_dpid': link.dst.dpid,
                    'src_port': link.src.port_no,
                    'dst_port': link.dst.port_no,
                    'bandwidth_utilization': 0.5,  # 默认值
                    'latency': 1.0,  # 默认延迟(ms)
                    'is_active': True
                })
        except Exception:
            # 如果无法获取拓扑，使用默认的树形拓扑
            links = self._get_default_tree_topology()
        
        return links
    
    def _get_default_tree_topology(self):
        """Retrieve the default tree topology (for testing)."""
        # Assume a 3-layer tree topology: 1 core + 2 aggregation + 4 edge
        # s1 (core) -> s2, s3 (aggregation)
        # s2 -> s4, s5 (edge)
        # s3 -> s6, s7 (edge)
        
        default_links = [
            {'src_dpid': 1, 'dst_dpid': 2, 'src_port': 1, 'dst_port': 1,
             'bandwidth_utilization': 0.3, 'latency': 1.0, 'is_active': True},
            {'src_dpid': 1, 'dst_dpid': 3, 'src_port': 2, 'dst_port': 1,
             'bandwidth_utilization': 0.3, 'latency': 1.0, 'is_active': True},
            {'src_dpid': 2, 'dst_dpid': 4, 'src_port': 2, 'dst_port': 1,
             'bandwidth_utilization': 0.2, 'latency': 1.0, 'is_active': True},
            {'src_dpid': 2, 'dst_dpid': 5, 'src_port': 3, 'dst_port': 1,
             'bandwidth_utilization': 0.2, 'latency': 1.0, 'is_active': True},
            {'src_dpid': 3, 'dst_dpid': 6, 'src_port': 2, 'dst_port': 1,
             'bandwidth_utilization': 0.2, 'latency': 1.0, 'is_active': True},
            {'src_dpid': 3, 'dst_dpid': 7, 'src_port': 3, 'dst_port': 1,
             'bandwidth_utilization': 0.2, 'latency': 1.0, 'is_active': True},
        ]
        
        return default_links
    
    def _get_node_features(self, switch_id, switches, links):
        """
        Get the feature vector of a single node.

        Features:
        - cpu_available: CPU availability [0, 1]
        - memory_available: Memory availability [0, 1]
        - has_ml_model: Whether an ML model is deployed {0, 1}
        - traffic_load: Traffic load [0, 1] (normalized)
        - attack_detected: Whether an attack is detected {0, 1}
        - degree: Node degree (number of connected links) normalized
        - is_edge_switch: Whether it is an edge switch {0, 1}
        """
        sw_state = switches.get(switch_id, {})
        
        # 基础特征
        cpu_available = sw_state.get('cpu_available', 1.0)
        memory_available = sw_state.get('memory_available', 1.0)
        has_ml_model = 1.0 if sw_state.get('has_ml_model', False) else 0.0
        traffic_load = min(sw_state.get('traffic_load', 0.0), 1.0)
        attack_detected = 1.0 if sw_state.get('attack_detected', False) else 0.0
        
        # 计算节点度数
        degree = sum(1 for link in links 
                     if link['src_dpid'] == switch_id or link['dst_dpid'] == switch_id)
        max_degree = max(1, max(
            sum(1 for link in links if link['src_dpid'] == sw or link['dst_dpid'] == sw)
            for sw in switches.keys()
        ) if switches else 1)
        normalized_degree = degree / max_degree
        
        # 判断是否是边缘交换机（度数较小且连接主机）
        is_edge = 1.0 if degree <= 2 else 0.0
        
        return np.array([
            cpu_available,
            memory_available, 
            has_ml_model,
            traffic_load,
            attack_detected,
            normalized_degree,
            is_edge
        ])
    
    def get_state_dim(self):
        """Return the node feature dimension."""
        return self.node_feature_dim
    
    def get_action_dim(self):
        """Return the action dimension (equal to the number of switches)."""
        return self.model_manager.num_switches


class GraphDataBuilder:
    """
    Graph Data Builder
    Converts graph states into PyTorch tensor format
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def build(self, node_features, edge_list):
        """
        Build graph data object.

        Args:
            node_features: Numpy array [num_nodes, feature_dim]
            edge_list: Edge list [(src, dst), ...]
        
        Returns:
            Graph data object
        """
        from networks.gnn import GraphData
        
        x = torch.FloatTensor(node_features).to(self.device)
        
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous().to(self.device)
        else:
            # 如果没有边，创建空的边索引
            edge_index = torch.LongTensor([[], []]).to(self.device)
        
        return GraphData(x=x, edge_index=edge_index)
    
    def batch(self, data_list):
        """Batch process multiple graphs."""
        from networks.gnn import GraphBatch
        return GraphBatch.from_data_list(data_list)

