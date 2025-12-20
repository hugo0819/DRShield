"""
SDN Environment
Supports two state representation modes:
1. Flat vector state (for traditional PPO/DDPG)
2. Graph state (for GNN+PPO)
"""

import random
from ryu.lib import hub
import requests
import json
import numpy as np

# Import from project files
import config
from managers.enhanced_state_space import EnhancedStateSpace
from managers.enhanced_action_space import EnhancedActionSpace
from managers.abstract_model_manager import AbstractModelManager


class SDNEnvironment:
    """
    SDN Reinforcement Learning Environment

    Supports two modes:
    - Traditional mode: Uses flat vector state
    - GNN mode: Uses graph structure state
    """

    def __init__(self, controller, num_switches, num_hosts, use_gnn=None):
        """
        Initialize SDN Environment

        Args:
            controller: SDN controller instance
            num_switches: Number of switches
            num_hosts: Number of hosts
            use_gnn: Whether to use GNN mode (None to read from config)
        """
        self.controller = controller
        self.num_switches = num_switches
        self.num_hosts = num_hosts

        # Determine whether to use GNN mode
        self.use_gnn = use_gnn if use_gnn is not None else getattr(config, 'USE_GNN_PPO', False)

        # Core components
        self.model_manager = AbstractModelManager(num_switches=self.num_switches)
        self.action_space = EnhancedActionSpace(num_switches=self.num_switches)

        # Select state space and reward calculator based on mode
        if self.use_gnn:
            from managers.graph_state_space import GraphStateSpace, GraphDataBuilder
            from managers.reward_function import NormalizedRewardCalculator
            from networks.gnn import GraphData

            self.graph_state_space = GraphStateSpace(
                self.model_manager, 
                controller, 
                device=config.DEVICE
            )
            self.graph_data_builder = GraphDataBuilder(device=config.DEVICE)
            self.reward_calculator = NormalizedRewardCalculator(self.model_manager)

            # Dimensions for GNN mode
            self.node_feature_dim = config.NODE_FEATURE_DIM
            self.action_dim = self.num_switches

            # Cache switch mappings
            self._switch_id_to_idx = None
            self._idx_to_switch_id = None
        else:
            from managers.simplified_reward_function import SimplifiedRewardCalculator

            self.state_space = EnhancedStateSpace(self.model_manager, controller)
            self.reward_calculator = SimplifiedRewardCalculator(self.model_manager)

            # Dimensions for traditional mode
            self.state_dim = len(self.state_space.get_enhanced_state())
            self.action_dim = self.action_space.num_switches

        # Statistics
        self.episode_count = 0
        self.total_rewards = []

    def reset(self):
        """
        Reset the environment

        Returns:
            state: Initial state (vector or graph data)
        """
        # Remove all deployments
        for switch_id in list(self.model_manager.deployed_switches):
            self.model_manager.remove_model(switch_id)

        # Update controller state
        self._update_controller_deployments()

        self.episode_count += 1

        if self.use_gnn:
            return self._get_graph_state()
        else:
            return self.state_space.get_enhanced_state()

    def step(self, action):
        """
        Execute action and return next state

        Args:
            action: Action vector (deployment decisions for each switch)

        Returns:
            next_state: Next state
            reward: Reward
            done: Whether the episode is done
        """
        # Execute action
        self._execute_action(action)

        # Trigger traffic generation
        self._run_traffic_episode()

        # Get next state
        if self.use_gnn:
            next_state = self._get_graph_state()
        else:
            next_state = self.state_space.get_enhanced_state()

        # Calculate reward
        current_deployments = self.model_manager.deployed_switches

        if self.use_gnn:
            # Use normalized reward function
            # Estimate detected attacks and false positives
            detected_attacks = int(self.controller.attack_count * 0.95) if current_deployments else 0
            false_positives = int(self.controller.benign_count * 0.01) if current_deployments else 0
            latency = len(current_deployments) * 2.0  # Simplified latency estimation

            reward = self.reward_calculator.calculate_reward(
                attack_packets=self.controller.attack_count,
                benign_packets=self.controller.benign_count,
                current_deployments=current_deployments,
                detected_attacks=detected_attacks,
                false_positives=false_positives,
                latency_ms=latency
            )
        else:
            reward = self.reward_calculator.calculate_reward(
                self.controller.attack_count, 
                self.controller.benign_count,
                current_deployments
            )

        self.total_rewards.append(reward)
        done = False

        return next_state, reward, done

    def _get_graph_state(self):
        """
        Get graph structure state

        Returns:
            GraphData: Graph data object
        """
        from networks.gnn import GraphData
        import torch

        # Get graph state
        node_features, edge_list, switch_id_to_idx, idx_to_switch_id = \
            self.graph_state_space.get_graph_state()

        # Cache mapping relationships
        self._switch_id_to_idx = switch_id_to_idx
        self._idx_to_switch_id = idx_to_switch_id

        # Build graph data
        graph_data = self.graph_data_builder.build(node_features, edge_list)

        return graph_data

    def _run_traffic_episode(self):
        """Send API request to start a traffic episode in Mininet."""
        api_url = f"{config.MININET_API_URL}/start_episode"
        headers = {'Content-Type': 'application/json'}

        # Randomly select attacker and normal user
        hosts_ids = list(range(min(8, self.num_hosts)))
        if len(hosts_ids) >= 2:
            attacker_id, benign_id = random.sample(hosts_ids, 2)
        else:
            attacker_id, benign_id = 0, 1

        payload = {
            'attacker_id': attacker_id,
            'benign_id': benign_id
        }

        try:
            print(f"Sending traffic generation request to Mininet API: {payload}")
            response = requests.post(api_url, data=json.dumps(payload), headers=headers, timeout=2)

            if response.status_code == 200:
                # Optimize traffic statistics timing:
                hub.sleep(3)
                self.controller.request_stats()
                hub.sleep(config.EPISODE_LENGTH - 3)
                print("Traffic episode ended.")
            else:
                print(f"Failed to start traffic: {response.status_code} - {response.text}")
                hub.sleep(config.EPISODE_LENGTH)
        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to Mininet API: {e}")
            hub.sleep(config.EPISODE_LENGTH)

    def _execute_action(self, action_vector):
        """Execute action vector."""
        changed = False

        for i, action in enumerate(action_vector):
            # In GNN mode, map index back to switch ID
            if self.use_gnn and self._idx_to_switch_id:
                switch_id = self._idx_to_switch_id.get(i, i + 1)
            else:
                switch_id = i + 1

            if action == 1 and not self.model_manager.is_deployed(switch_id):
                if self.model_manager.deploy_model(switch_id):
                    print(f"Deploying model on switch {switch_id}")
                    changed = True
            elif action == 0 and self.model_manager.is_deployed(switch_id):
                if self.model_manager.remove_model(switch_id):
                    print(f"Removing model from switch {switch_id}")
                    changed = True

        if changed:
            self._update_controller_deployments()

    def _update_controller_deployments(self):
        """Update the deployment list in the controller's memory."""
        deployed_dpid_list = list(self.model_manager.deployed_switches)
        self.controller.deployed_ml_switches = set(deployed_dpid_list)
        print(f"Updated controller deployment status: {self.controller.deployed_ml_switches}")

    def get_average_reward(self, last_n=100):
        """Get the average reward for the last N episodes."""
        if not self.total_rewards:
            return 0.0
        recent = self.total_rewards[-last_n:]
        return sum(recent) / len(recent)
