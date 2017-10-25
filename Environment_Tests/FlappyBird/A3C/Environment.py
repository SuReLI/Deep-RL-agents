
import os

from ple.games.flappybird import FlappyBird
from ple import PLE

import cv2
import numpy as np
from collections import deque

from parameters import ENV, FRAME_SKIP, FRAME_BUFFER_SIZE

from PIL import Image
import imageio


class Environment:

    def __init__(self, worker_index=0):

        self.env = PLE(FlappyBird(), fps=30, display_screen=False)

        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.reset()

        self.offset = 118

        self.render = False
        self.images = []
        self.tmp_name = 'tmp_{}.png'.format(worker_index)

    def get_state_size(self):
        return (47, 47, FRAME_BUFFER_SIZE)

    def get_action_size(self):
        return 2

    def set_render(self, render):
        self.env.display_screen = render

    def _convert_process_buffer(self):
        # Convert RGB images into grayscale images
        state = map(lambda x: cv2.resize(x, (47, 47))[:, :, np.newaxis],
                    self.frame_buffer)

        # Concatenate the frames to get a single state
        return np.concatenate(list(state), axis=2)

    def reset(self):
        # Reset the frame buffer with FRAME_BUFFER_SIZE frames
        self.env.reset_game()

        for _ in range(FRAME_BUFFER_SIZE):
            self.frame_buffer.append(self.env.getScreenGrayscale().T)
            self.env.act(0)

        return self._convert_process_buffer()

    def act(self, action, gif=False):
        action += self.offset
        r = 0
        i, done = 0, False
        while i < (FRAME_SKIP + 1) and not done:
            # If gif, save image
            if gif:
                self.env.saveScreen(self.tmp_name)
                self.images.append(imageio.imread(self.tmp_name))

            r_tmp = self.env.act(action)
            done = self.env.game_over()
            r += r_tmp
            i += 1

        self.frame_buffer.append(self.env.getScreenGrayscale())

        return self._convert_process_buffer(), r, done, ""

    def save_gif(self, path, i):
        path = path + "_{}.gif".format(i)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

    def close(self):
        self.env.display_screen = False
