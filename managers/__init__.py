"""
SDN DDoS防御系统 - 管理器模块

包含:
- AbstractModelManager: 抽象模型管理器
- EnhancedStateSpace: 增强状态空间（扁平向量）
- EnhancedActionSpace: 增强动作空间
- GraphStateSpace: 图状态空间（用于GNN）
- SimplifiedRewardCalculator: 简化奖励计算器
- NormalizedRewardCalculator: 归一化奖励计算器（MR/FPR/RU/LTD）
"""

from .abstract_model_manager import AbstractModelManager
from .enhanced_state_space import EnhancedStateSpace
from .enhanced_action_space import EnhancedActionSpace
from .simplified_reward_function import SimplifiedRewardCalculator

# 可选导入（GNN相关）
try:
    from .graph_state_space import GraphStateSpace, GraphDataBuilder
    from .reward_function import NormalizedRewardCalculator
except ImportError:
    pass

__all__ = [
    'AbstractModelManager',
    'EnhancedStateSpace',
    'EnhancedActionSpace',
    'SimplifiedRewardCalculator',
    'GraphStateSpace',
    'GraphDataBuilder',
    'NormalizedRewardCalculator'
]

