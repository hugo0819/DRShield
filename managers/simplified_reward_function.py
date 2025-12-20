"""
Simplified Reward Function
For traditional PPO/DDPG modes (non-GNN modes).
"""

import numpy as np


class SimplifiedRewardCalculator:
    """Simplified reward calculator (for traditional modes)."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.previous_deployments = set()
        self.efficiency_weight = 1.0

    def calculate_reward(self, attack_packets, benign_packets, current_deployments):
        """
        Calculate reward.

        Args:
            attack_packets: Number of attack packets
            benign_packets: Number of benign packets
            current_deployments: Set of currently deployed switches
        """
        # 1. Detection effectiveness reward
        detection_reward = self._calculate_detection_reward(
            attack_packets, benign_packets, current_deployments)

        # 2. Resource efficiency reward
        efficiency_reward = self._calculate_efficiency_reward(current_deployments)

        # 3. Deployment strategy reward
        strategy_reward = self._calculate_strategy_reward(current_deployments)

        total_reward = (0.5 * detection_reward + 
                       0.3 * efficiency_reward + 
                       0.2 * strategy_reward)

        self.previous_deployments = current_deployments.copy()
        return total_reward

    def _calculate_detection_reward(self, attack_packets, benign_packets, deployments):
        """Calculate reward based on detection effectiveness."""
        if attack_packets == 0:
            return 0.1  # Base reward when there are no attacks

        # If at least one switch in the set successfully detects, consider detection successful
        detection_triggered = False
        for _ in deployments:
            if np.random.random() < 0.95:
                detection_triggered = True
                break

        return 1.0 if detection_triggered else 0.0

    def _calculate_efficiency_reward(self, deployments):
        """
        Calculate efficiency reward.

        Args:
            deployments (set): Set of switches with deployed models
        """
        if self.model_manager.num_switches == 0:
            return 0

        num_deployments = len(deployments)
        deployment_ratio = num_deployments / self.model_manager.num_switches

        # The fewer deployments, the higher the reward
        efficiency_reward = 1.0 - deployment_ratio
        return efficiency_reward * self.efficiency_weight

    def _calculate_strategy_reward(self, deployments):
        """Calculate reward for strategy rationality."""
        edge_switches = {1, 2, 3}
        edge_deployments = deployments.intersection(edge_switches)

        if not edge_switches:
            return 0.0

        # Reward is proportional to the deployment ratio on edge switches
        strategy_score = len(edge_deployments) / len(edge_switches)
        return 0.2 * strategy_score

