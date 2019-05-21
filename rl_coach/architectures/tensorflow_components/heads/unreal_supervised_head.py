
import tensorflow as tf
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition


class UnrealSupervisedHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        # which supervised loss will be considered?
        self.