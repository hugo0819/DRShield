import numpy as np
import torch

class AbstractModelManager:
    """Abstract Model Manager - Used during reinforcement learning training phase."""
    
    def __init__(self, num_switches=7, detection_rate=0.95):
        self.num_switches = num_switches
        self.detection_rate = detection_rate  # Achieved detection rate
        
        # Abstract model specifications
        self.model_specs = {
            'cpu_cost': 0.3,      # CPU resource consumption ratio
            'memory_cost': 0.25,  # Memory resource consumption ratio
            'latency': 2.0,       # Detection latency (ms)
            'accuracy': 0.95      # Achieved detection rate
        }
        
        # Switch state tracking
        self.switch_states = {}
        self.deployed_switches = set()  # Switches with deployed ML models
        
        self._initialize_switches()
    
    def _initialize_switches(self):
        """Initialize switch states."""
        for switch_id in range(1, self.num_switches + 1):
            self.switch_states[switch_id] = {
                'cpu_available': 1.0,
                'memory_available': 1.0,
                'has_ml_model': False,
                'traffic_load': 0.0,
                'attack_detected': False
            }
    
    def can_deploy_model(self, switch_id):
        """Check if a model can be deployed on the specified switch."""
        if switch_id not in self.switch_states:
            return False
            
        state = self.switch_states[switch_id]
        cpu_enough = state['cpu_available'] >= self.model_specs['cpu_cost']
        memory_enough = state['memory_available'] >= self.model_specs['memory_cost']
        
        return cpu_enough and memory_enough and not state['has_ml_model']
    
    def is_deployed(self, switch_id):
        """Check if a model is already deployed on the specified switch."""
        return switch_id in self.deployed_switches

    def deploy_model(self, switch_id):
        """Abstract model deployment (does not actually deploy)."""
        if not self.can_deploy_model(switch_id):
            return False
            
        # Update resource usage
        state = self.switch_states[switch_id]
        state['cpu_available'] -= self.model_specs['cpu_cost']
        state['memory_available'] -= self.model_specs['memory_cost']
        state['has_ml_model'] = True
        
        self.deployed_switches.add(switch_id)
        return True
    
    def remove_model(self, switch_id):
        """Remove model deployment."""
        if switch_id not in self.switch_states or not self.switch_states[switch_id]['has_ml_model']:
            return False
            
        # Release resources
        state = self.switch_states[switch_id]
        state['cpu_available'] += self.model_specs['cpu_cost']
        state['memory_available'] += self.model_specs['memory_cost']
        state['has_ml_model'] = False
        
        self.deployed_switches.discard(switch_id)
        return True
    
    def simulate_detection(self, switch_id, has_attack=False):
        """Simulate the detection process."""
        if switch_id not in self.deployed_switches:
            return False  # No model deployed, cannot detect
            
        if not has_attack:
            return False  # No attack, correctly return False
            
        # Simulate based on the 95% detection rate
        return np.random.random() < self.detection_rate