
import os
import gym
import cv2
import numpy as np
from collections import deque

import game.wrapped_flappy_bird as game

from parameters import ENV, FRAME_SKIP, FRAME_BUFFER_SIZE

from PIL import Image
import imageio


def onehot(action):
    action_onehot = [0] * 2
    action_onehot[action] = 1
    return action_onehot


class Environment:

    def __init__(self, worker_index=0):

        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.reset()

        self.render = False
        self.images = []
        self.tmp_name = 'tmp_{}.png'.format(worker_index)

    def get_state_size(self):
        return (84, 84, FRAME_BUFFER_SIZE)

    def get_action_size(self):
        return 2

    def set_render(self, render):
        self.render = render

    def _convert_process_buffer(self):
        # Convert RGB images into grayscale images
        state = map(lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
                                         (84, 90)),
                    self.frame_buffer)

        # Cut the images to get a 84x84 image
        state = map(lambda x: x[1:85, :, np.newaxis], state)

        # Concatenate the frames to get a single state
        return np.concatenate(list(state), axis=2)

    def reset(self):
        self.env = game.GameState(1, False)

        # Reset the frame buffer with FRAME_BUFFER_SIZE frames
        for _ in range(FRAME_BUFFER_SIZE):
            frame, r, done = self.env.frame_step(onehot(0))
            self.frame_buffer.append(frame)

        return self._convert_process_buffer()

    def act(self, action, gif=False):
        if not gif:
            return self._act(action)
        else:
            return self._act_gif(action)

    def _act(self, action):

        s, r, done = self.env.frame_step(onehot(action))
        self.frame_buffer.append(s)
        return self._convert_process_buffer(), r, done, ""

    def _act_gif(self, action):

        r = 0
        i, done = 0, False
        while i < (FRAME_SKIP + 1) and not done:

            s_, r_tmp, done = self.env.frame_step(onehot(action))
            r += r_tmp
            i += 1

            # Save image
            img = Image.fromarray(s_)
            img.save(self.tmp_name)
            self.images.append(imageio.imread(self.tmp_name))

        self.frame_buffer.append(s_)
        return self._convert_process_buffer(), r, done, ""

    def save_gif(self, path, i):
        path = path + "_{}.gif".format(i)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

    def close(self):
        pass
