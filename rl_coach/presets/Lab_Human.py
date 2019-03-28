from rl_coach.environments.lab_environment import LabEnvironment, LabEnvironmentParameters, level_scripts
from rl_coach.agents.human_agent import HumanAgentParameters
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.graph_managers.graph_manager import SimpleSchedule, EnvironmentSteps, EnvironmentEpisodes, TrainingSteps, ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.base_parameters import VisualizationParameters
# agent
agent_params = HumanAgentParameters()

# environment
env_params = LabEnvironmentParameters(level=SingleLevelSelection(levels=level_scripts), rotation=20, human_control=False, fps=30)

# scheduler param
schedule_params = SimpleSchedule()
# schedule_params = ScheduleParameters()
schedule_params.heatup_steps = EnvironmentSteps(20)

vis_params = VisualizationParameters(render=True)

# graph manager
graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=vis_params
)
