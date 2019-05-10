import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.environments.lab_environment import LabEnvironment
from rl_coach.base_parameters import VisualizationParameters
import numpy as np
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, ImageObservationSpace, VectorObservationSpace


@pytest.fixture()
def lab_env():
    # create a breakout gym environment
    env = LabEnvironment(level='nav_maze_static_01', seed=10, frame_skip=4, human_control=False, rotation=20, width=84, height=84, fps=60, custom_reward_threshold=None,
                         visualization_parameters=VisualizationParameters(), random_initialization_steps=30)
    return env



@pytest.mark.unit_test
def test_lab_discrete_environment(lab_env):
    # observation space
    assert type(lab_env.state_space['observation']) == ImageObservationSpace
    assert np.all(lab_env.state_space['observation'].shape == [84, 84, 3])
    assert np.all(lab_env.last_env_response.next_state['observation'].shape == (84, 84, 3))

    assert np.all(lab_env.state_space['depth'].shape == [84, 84, 1])
    # action space
    assert type(lab_env.action_space) == DiscreteActionSpace
    assert np.all(lab_env.action_space.high == 6)






@pytest.mark.unit_test
def test_step(lab_env):
    result = lab_env.step(0)

if __name__ == '__main__':
    test_lab_discrete_environment(lab_env())