import numpy as np

class EnhancedStateSpace:
    def __init__(self, model_manager, traffic_monitor):
        self.model_manager = model_manager
        self.traffic_monitor = traffic_monitor
        self.num_switches = model_manager.num_switches
        # Each switch has 21 features: 15 traffic + 4 resources + 2 deployment
        self.state_dim_per_switch = 21

    def get_enhanced_state(self):
        """Get the enhanced state vector for the entire network."""
        full_state = []
        for switch_id in range(1, self.num_switches + 1):
            switch_state = self._get_switch_state(switch_id)
            full_state.extend(switch_state)
        return np.array(full_state)

    def _get_switch_state(self, switch_id):
        """Get the state features of a single switch."""
        # ✅ Add missing method calls
        traffic_features = self._get_traffic_features(switch_id)
        resource_features = self._get_resource_features(switch_id)
        deployment_features = self._get_deployment_features(switch_id)

        return traffic_features + resource_features + deployment_features

    # ✅ New: Implement missing methods
    def _get_traffic_features(self, switch_id):
        """Get traffic-related features (15 features)."""
        state = self.traffic_monitor.state.get(switch_id, [])
        features = []

        if not state:
            return [0] * 15

        port_data = state[0]
        # Port features (3 ports * 4 features = 12)
        for port_no in range(1, 4):
            port_stats = port_data.get(port_no, [0, 0, 0, 0])
            features.extend([int(v) for v in port_stats])

        # Overall traffic features for the switch (3 features)
        features.append(state[1])  # packet_count
        features.append(state[2])  # byte_count
        features.append(state[3])  # flow_count

        return features

    def _get_resource_features(self, switch_id):
        """Get resource-related features (4 features)."""
        switch_info = self.model_manager.switch_states.get(switch_id, {})
        return [
            switch_info.get('cpu_available', 1.0),
            switch_info.get('memory_available', 1.0),
            # Simplified traffic load
            self.traffic_monitor.state.get(switch_id, [{}, 0, 0, 0])[2] / 1000000.0,
            1.0 if switch_info.get('has_ml_model', False) else 0.0
        ]

    def _get_deployment_features(self, switch_id):
        """Get deployment-related features (2 features)."""
        switch_info = self.model_manager.switch_states.get(switch_id, {})
        return [
            1.0 if switch_info.get('has_ml_model', False) else 0.0,
            switch_info.get('attack_detected', 0.0)
        ]