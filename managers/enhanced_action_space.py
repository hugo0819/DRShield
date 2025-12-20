class EnhancedActionSpace:
    """Enhanced action space - includes bandwidth allocation and model deployment decisions."""
    
    def __init__(self, num_switches=7):
        self.num_switches = num_switches
        # Each switch: Bandwidth allocation (1) + Model deployment decision (1)
        # self.action_dim = num_switches * 2
        # Model deployment decision (1)
        self.action_dim = num_switches
    
    def decode_actions(self, action_vector):
        """Decode action vector."""
        actions = {}
        
        for i in range(self.num_switches):
            switch_id = i + 1
            # base_idx = i * 2
            
            # # Bandwidth allocation (0-1, mapped to actual bandwidth)
            # bandwidth_ratio = np.clip(action_vector[base_idx], 0, 1)
            # bandwidth = bandwidth_ratio * 10000  # Map to range 0-10000
            
            # Model deployment decision (-1 to 1, >0 means deploy, <0 means remove)
            deployment_decision = action_vector[i]
            
            actions[switch_id] = {
                # 'bandwidth': bandwidth,
                'deploy_model': deployment_decision > 0
            }
        
        return actions