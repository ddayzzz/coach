from rl_coach.agents.human_agent import HumanAgentParameters
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
# from rl_coach.environments.environment import SingleLevelSelection
# from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.lab_environment import LabEnvironmentParameters, level_scripts
####################
# Graph Scheduling #
####################
schedule_params = SimpleSchedule()
# schedule_params.heatup_steps = EnvironmentSteps(10000)

#########
# Agent #
#########
agent_params = HumanAgentParameters()

###############
# Environment #
###############
env_params = LabEnvironmentParameters(level=SingleLevelSelection(level_scripts), human_control=True, width=100, height=100)


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(render=True, native_rendering=True))
