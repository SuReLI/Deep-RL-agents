# -*- coding: utf-8 -*-
import os
import numpy as np
from ale_python_interface import ALEInterface
import cv2

from settings import *


class Environment:

    def __init__(self, render=False):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 0)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setInt(b'frame_skip', 4)
        self.ale.setBool(b'display_screen', render)
        self.ale.loadROM(ENV.encode('ascii'))
        self._screen = np.empty((210, 160, 1), dtype=np.uint8)
        self._no_op_max = 7

    def set_render(self, render):
        if not render:
            self.ale.setBool(b'display_screen', render)

    def reset(self):
        self.ale.reset_game()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.ale.act(0)

        self.ale.getScreenGrayscale(self._screen)
        screen = np.reshape(self._screen, (210, 160))
        screen = cv2.resize(screen, (84, 110))
        screen = screen[18:102, :]
        screen = screen.astype(np.float32)
        screen /= 255.0

        self.frame_buffer = np.stack((screen, screen, screen, screen), axis=2)
        return self.frame_buffer

    def process(self, action):

        reward = self.ale.act(4+action)
        done = self.ale.game_over()

        self.ale.getScreenGrayscale(self._screen)
        screen = np.reshape(self._screen, (210, 160))
        screen = cv2.resize(screen, (84, 110))
        screen = np.reshape(screen[18:102, :], (84, 84, 1))
        screen = screen.astype(np.float32)
        screen *= (1/255.0)

        self.frame_buffer = np.append(self.frame_buffer[:, :, 1:],
                                      screen, axis=2)

        return self.frame_buffer, reward, done, ""

    def close(self):
        self.ale.setBool(b'display_screen', False)
