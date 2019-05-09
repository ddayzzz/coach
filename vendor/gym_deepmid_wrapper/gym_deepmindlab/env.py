import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from pygame import locals
import deepmind_lab

LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
          'nav_maze_random_goal_01', 'nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_static_01', \
          'nav_maze_static_02', 'nav_maze_static_03', 'seekavoid_arena_01', 'stairway_to_melon']


def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('_')))


MAP = {_to_pascal(l): l for l in LEVELS}


class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, scene, colors='RGB_INTERLEAVED', width=84, height=84, **kwargs):
        super(DeepmindLabEnv, self).__init__(**kwargs)

        if not scene in LEVELS:
            raise Exception('Scene %s not supported' % (scene))

        self._colors = colors
        self._lab = deepmind_lab.Lab(scene, [self._colors], dict(fps=str(60), width=str(width), height=str(height)))

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)

        self._last_observation = None
        self.viewer = None

    def step(self, action):
        reward = self._lab.step(ACTION_LIST[action], num_steps=4)
        terminal = not self._lab.is_running()
        obs = None if terminal else self._lab.observations()[self._colors]
        self._last_observation = obs if obs is not None else np.copy(self._last_observation)
        return self._last_observation, reward, terminal, dict()

    def reset(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._lab.reset()
        self._last_observation = self._lab.observations()[self._colors]
        return self._last_observation

    def seed(self, seed=None):
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        rgb_img = self._lab.observations()[self._colors]
        if mode == 'rgb_array':
            return rgb_img
        elif mode is 'human':
            # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(rgb_img)
            return self.viewer.isopen
        else:
            super(DeepmindLabEnv, self).render(mode=mode)  # just raise an exception

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in range(6)]

    def get_keys_to_action(self):

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
    _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
    _action(20, 0, 0, 0, 0, 0, 0),  # look_right
    # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
    # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
    _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
    _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
    _action(0, 0, 0, 1, 0, 0, 0),  # forward
    _action(0, 0, 0, -1, 0, 0, 0),  # backward
    # _action(  0,   0,  0,  0, 1, 0, 0), # fire
    # _action(  0,   0,  0,  0, 0, 1, 0), # jump
    # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
]

ACTION_MEANING = {
    0: "LOOK_LEFT",
    1: "LOOK_RIGHT",
    2: "STRAFE_LEFT",
    3: "STRAFE_RIGHT",
    4: "FORWARD",
    5: "BACKWARD"
}

KEYWORD_TO_KEY = {
    'LOOK_LEFT': locals.K_4,
    'LOOK_RIGHT': locals.K_6,
    'STRAFE_LEFT': locals.K_LEFT,
    'STRAFE_RIGHT': locals.K_RIGHT,
    'FORWARD': locals.K_UP,
    'BACKWARD': locals.K_DOWN
}
