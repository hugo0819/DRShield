import numpy as np
import config

class NormalizedRewardCalculator:
    """
    Normalized reward calculator based on four metrics:
    - MR (Mitigation Rate): Benefit-oriented
    - FPR (False Positive Rate): Cost-oriented
    - RU (Resource Utilization): Cost-oriented
    - LTD (Latency): Cost-oriented
    """

    def __init__(self, model_manager, 
                 w1=0.4, w2=0.2, w3=0.2, w4=0.2,  # Weights
                 T_FPR=0.05,    # FPR tolerance upper limit (5%)
                 tau=10.0,      # LTD expected latency (ms)
                 alpha=20.0):   # FPR exponential decay coefficient
        self.model_manager = model_manager

        # Weights for the four metrics
        self.w1 = w1  # MR weight
        self.w2 = w2  # FPR weight
        self.w3 = w3  # RU weight
        self.w4 = w4  # LTD weight

        # Normalization parameters
        self.T_FPR = T_FPR
        self.tau = tau
        self.alpha = alpha

    def calculate_reward(self, attack_packets, benign_packets, 
                         current_deployments, detected_attacks=0, 
                         false_positives=0, latency_ms=0.0):
        """
        Calculate the total normalized reward.

        Args:
            attack_packets: Total number of attack packets
            benign_packets: Total number of benign packets
            current_deployments: Set of currently deployed switches
            detected_attacks: Number of successfully detected attack packets
            false_positives: Number of falsely reported benign packets
            latency_ms: Detection latency (in milliseconds)
        """
        # 1. MR (Mitigation Rate) - Benefit-oriented
        # f_MR(x) = x, raw value as the ratio
        MR = self._calculate_mitigation_rate(attack_packets, detected_attacks)
        f_MR = MR  # No transformation needed

        # 2. FPR (False Positive Rate) - Cost-oriented
        # f_FPR(x) = e^(-alpha * x) or 1 - x/T_FPR
        FPR = self._calculate_false_positive_rate(benign_packets, false_positives)
        f_FPR = self._normalize_fpr(FPR)

        # 3. RU (Resource Utilization) - Cost-oriented
        # f_RU(x) = 1 - x, representing "remaining resource ratio"
        RU = self._calculate_resource_utilization(current_deployments)
        f_RU = 1.0 - RU  # Remaining resource ratio

        # 4. LTD (Latency) - Cost-oriented
        # f_LTD(x) = e^(-x/tau)
        f_LTD = self._normalize_latency(latency_ms)

        # Weighted sum
        total_reward = (self.w1 * f_MR + 
                       self.w2 * f_FPR + 
                       self.w3 * f_RU + 
                       self.w4 * f_LTD)

        return total_reward

    def _calculate_mitigation_rate(self, attack_packets, detected_attacks):
        """Calculate mitigation rate MR = detected / total_attacks."""
        if attack_packets == 0:
            return 1.0  # No attacks, perfect mitigation
        return min(detected_attacks / attack_packets, 1.0)

    def _calculate_false_positive_rate(self, benign_packets, false_positives):
        """Calculate false positive rate FPR = false_positives / benign_packets."""
        if benign_packets == 0:
            return 0.0  # No benign traffic, no false positives
        return min(false_positives / benign_packets, 1.0)

    def _normalize_fpr(self, fpr):
        """
        Normalize FPR using exponential decay:
        f_FPR(x) = e^(-alpha * x)
        - When x=0, the score is 1 (perfect)
        - The larger the penalty, the smoother the gradient
        """
        return np.exp(-self.alpha * fpr)

    def _calculate_resource_utilization(self, current_deployments):
        """Calculate resource utilization RU = deployed / total_switches."""
        num_switches = self.model_manager.num_switches
        if num_switches == 0:
            return 0.0
        return len(current_deployments) / num_switches

    def _normalize_latency(self, latency_ms):
        """
        Normalize latency using exponential decay:
        f_LTD(x) = e^(-x/tau)
        - When x=tau, the score is approximately 0.37
        - Maps [0, ∞) to (0, 1]
        """
        return np.exp(-latency_ms / self.tau)