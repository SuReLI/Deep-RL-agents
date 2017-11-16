# -*- coding: utf-8 -*-
import os
import numpy as np
from ale_python_interface import ALEInterface
import cv2
import imageio

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
        self._screen = np.empty((250, 160, 1), dtype=np.uint8)
        self._no_op_max = 7

        self.img_buffer = []

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

        self.img_buffer = []
        self.img_buffer.append(self.ale.getScreenRGB())

        self.ale.getScreenGrayscale(self._screen)
        screen = np.reshape(self._screen, (250, 160))
        screen = cv2.resize(screen, (84, 125))
        screen = screen[15:99, :]
        screen = screen.astype(np.float32)
        screen /= 255.0

        self.frame_buffer = np.stack((screen, screen, screen, screen), axis=2)
        return self.frame_buffer

    def process(self, action, gif=False):

        reward = self.ale.act(1+action)
        done = self.ale.game_over()

        if gif:
            self.img_buffer.append(self.ale.getScreenRGB())

        self.ale.getScreenGrayscale(self._screen)
        screen = np.reshape(self._screen, (250, 160))
        screen = cv2.resize(screen, (84, 125))
        screen = np.reshape(screen[15:99, :], (84, 84, 1))
        screen = screen.astype(np.float32)
        screen *= (1/255.0)

        self.frame_buffer = np.append(self.frame_buffer[:, :, 1:],
                                      screen, axis=2)

        return self.frame_buffer, reward, done, ""

    def save_gif(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.img_buffer, duration=0.001)
        self.img_buffer = []

    def close(self):
        self.ale.setBool(b'display_screen', False)
