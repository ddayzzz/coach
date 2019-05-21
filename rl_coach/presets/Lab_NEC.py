from rl_coach.agents.nec_agent import NECAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.environments.lab_environment import LabEnvironmentParameters
from rl_coach.environments.lab_environment import level_scripts, LabInputFilter

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(10000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(2000)

#########
# Agent #
#########
agent_params = NECAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00001
agent_params.input_filter = LabInputFilter()

###############
# Environment #
###############

env_params = LabEnvironmentParameters(level=SingleLevelSelection(level_scripts), human_control=False, width=84, height=84)

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test_using_a_trace_test = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
