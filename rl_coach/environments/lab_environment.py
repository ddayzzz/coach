try:
    import deepmind_lab
except ImportError:
    from rl_coach.logger import failed_imports

    failed_imports.append("deepmind_lab")

import os
import gym
import random
from enum import Enum
from os import path, environ
from typing import Union, List

import numpy as np

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import DiscreteActionSpace
from rl_coach.filters.action.full_discrete_action_space_map import FullDiscreteActionSpaceMap
from rl_coach.filters.filter import InputFilter, OutputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.filters.filter import NoOutputFilter, NoInputFilter
from rl_coach.spaces import MultiSelectActionSpace, ImageObservationSpace, VectorObservationSpace, StateSpace, ActionType
from pygame import locals


level_scripts = ['nav_maze_random_goal_03',
                 'nav_maze_random_goal_02',
                 'nav_maze_random_goal_01',
                 'nav_maze_static_03',
                 'nav_maze_static_02',
                 'nav_maze_static_01']

# filter for lab environment
##  process the information passed into the agent from the environment.
# LabInputFilter = NoInputFilter()
LabInputFilter = InputFilter(is_a_reference_filter=True)
# LabInputFilter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
LabInputFilter.add_observation_filter('observation', 'rescaling',
                                      ObservationRescaleToSizeFilter(
                                          ImageObservationSpace(np.array([84, 84, 3]), high=255)))
LabInputFilter.add_observation_filter('observation', 'to_grayscale', ObservationRGBToYFilter())
LabInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
# # stack last 4 images as s_t
LabInputFilter.add_observation_filter('observation', 'stacking', ObservationStackingFilter(4))
## what will we get from agent
LabOutputFilter = NoOutputFilter()

"""
lab environment parameters
"""


class LabEnvironmentParameters(EnvironmentParameters):

    def __init__(self, level, human_control=False, random_initialization_steps=30, rotation=20, width=84, height=84, fps=60):
        super().__init__(level=level)
        self.frame_skip = 4
        self.random_initialization_steps = random_initialization_steps
        self.default_input_filter = LabInputFilter
        self.default_output_filter = LabOutputFilter
        self.rotation = rotation
        self.human_control = human_control
        self.width = width
        self.height = height
        self.fps = fps


    @property
    def path(self):
        return 'rl_coach.environments.lab_environment:LabEnvironment'


class LabEnvironment(Environment):

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 rotation: int,
                 width: int,
                 height: int,
                 fps: int,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 random_initialization_steps: int,
                 target_success_rate: float = 1.0,
                 **kwargs):
        super(LabEnvironment, self).__init__(level=level, seed=seed, frame_skip=frame_skip, human_control=human_control,
                                             custom_reward_threshold=custom_reward_threshold,
                                             visualization_parameters=visualization_parameters,
                                             target_success_rate=target_success_rate)
        # other properties
        self.target_success_rate = target_success_rate
        self.random_initialization_steps = random_initialization_steps
        self.last_depth = None
        self.last_observation = None
        # deepmind lab environment
        ## environment
        avaiable_observations = ['RGB_INTERLEAVED', 'RGBD_INTERLEAVED']

        self.lab = deepmind_lab.Lab(self.env_id, avaiable_observations, config={'fps': str(fps), 'width': str(width), 'height': str(height)})
        ## action spec
        self.action_mapping, self.action_description, self.action_key_mapping = self._get_action_info(rotation=rotation)
        self.action_space = DiscreteActionSpace(num_actions=len(self.action_mapping),
                                                descriptions=self.action_description)
        self.key_to_action = self.action_key_mapping
        ## state spec


        rgb_obs = list(filter(lambda x: x['name'] == 'RGB_INTERLEAVED', self.lab.observation_spec()))[0]  # to get the size info

        self.state_space = StateSpace(
            {'depth': ImageObservationSpace(shape=np.hstack((rgb_obs['shape'][:2], [1])), high=255)})
        self.state_space['observation'] = ImageObservationSpace(shape=rgb_obs['shape'], high=255)
        # reset
        self.seed = seed if seed is not None else random.seed()
        #
        # self.lab.reset()
        self.reset_internal_state(True)
        # render
        # self.native_rendering = True  # from rl_coach.renderer import Renderer  # may be you can define your own render class
        if self.is_rendered:
            image = self.get_rendered_image()
            scale = 1
            if self.human_control:
                scale = 2
            if not self.native_rendering:
                self.renderer.create_screen(image.shape[1] * scale, image.shape[0] * scale)







    @staticmethod
    def _action(*entries):
        return np.array(entries, dtype=np.intc)

    def _get_action_info(self, rotation):
        """
        deepmind lab discrete action indices to vector
        :param rotation:
        :return:
        """

        action_list = [
            self._action(0, 0, 0, 0, 0, 0, 0),  # no op
            self._action(-rotation, 0, 0, 0, 0, 0, 0),  # look_left
            self._action(rotation, 0, 0, 0, 0, 0, 0),  # look_right
            # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
            # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
            self._action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
            self._action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
            self._action(0, 0, 0, 1, 0, 0, 0),  # forward
            self._action(0, 0, 0, -1, 0, 0, 0),  # backward
            # _action(  0,   0,  0,  0, 1, 0, 0), # fire
            # _action(  0,   0,  0,  0, 0, 1, 0), # jump
            # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
        ]
        description_list = ["NOOP", "LOOK_LEFT", "LOOK_RIGHT", "STRAFE_LEFT", "STRAFE_RIGHT", "FORWARD", "BACKWARD"]
        key_list = {
            (locals.K_LEFT,): 1,
            (locals.K_RIGHT,): 2,
            (locals.K_4,): 3,
            (locals.K_6,): 4,
            (locals.K_UP,): 5,
            (locals.K_DOWN,): 6
        }
        return action_list, description_list, key_list

    def get_rendered_image(self):
        rgb = self.state['observation']
        # d = self.state['depth']
        return rgb

    def _take_action(self, action: ActionType):
        reward = self.lab.step(action=self.action_mapping[action], num_steps=self.frame_skip)
        # print(self.done)
        # obs = self.lab.observations()
        #
        self.reward = reward
        #
        # # print('REWARD:', reward, 'DONE:', self.done,' ACTION:', action)
        # if self.done:
        #     # sinc step update state, done, reward
        #     # self.reset_internal_state(force_environment_reset=True)
        #     self.state['observation'] = self.last_observation
        #     self.state['depth'] = self.last_depth
        # else:
        #
        #     self.state['observation'] = obs['RGB_INTERLEAVED']
        #     self.state['depth'] = obs['RGBD_INTERLEAVED'][:, :, -1]



    def _is_running(self):
        return self.lab.is_running()

    def _update_state(self):
        """
        update the state
        :return:
        """
        self.done = not self._is_running()
        if self.done:
            # sinc step update state, done, reward
            # self.reset_internal_state(force_environment_reset=True)
            self.state['observation'] = self.last_observation
            self.state['depth'] = self.last_depth
        else:
            obs = self.lab.observations()
            self.state['observation'] = obs['RGB_INTERLEAVED']
            self.state['depth'] = obs['RGBD_INTERLEAVED'][:, :, -1]



    def _restart_environment_episode(self, force_environment_reset=False):
        self.lab.reset(seed=self.seed)

        self._random_noop()
        obs = self.lab.observations()
        self.last_observation = obs['RGB_INTERLEAVED']
        self.last_depth = obs['RGBD_INTERLEAVED'][:, :, -1]


    def get_target_success_rate(self):
        return self.target_success_rate

    def close(self):
        self.lab.close()

    def _random_noop(self):
        # simulate a random initial environment state by stepping for a random number of times between 0 and 30
        step_count = 0
        random_initialization_steps = random.randint(1, self.random_initialization_steps)
        while self.action_space is not None and (self.state is None or step_count < random_initialization_steps):
            step_count += 1
            self.step(self.action_space.default_action)






