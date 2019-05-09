# coding=utf-8
# render for deepmind lab to show
from rl_coach.renderer import Renderer
import pygame


class Window(object):

    def __init__(self, text_name):
        self.font = pygame.font.SysFont(None, 20)

    def process(self, *args, **kwargs):
        pass


class LabRender(Renderer):

    def __init__(self, enable_depth: bool=False, enable_v: bool=False, enable_pi: bool=False, enable_maze: bool=False):
        super(LabRender, self).__init__()
        self.enable_depth = enable_depth
        self.enable_v = enable_v
        self.enable_pi = enable_pi
        self.enable_maze = enable_maze
        #
        self._windows = {}

    def create_screen(self, width, height):
        """
        create serial windows to show those values
        :param width:
        :param height:
        :return:
        """
