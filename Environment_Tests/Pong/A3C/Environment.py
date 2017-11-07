
import os
import gym
import cv2
import numpy as np
from collections import deque

from parameters import ENV, FRAME_SKIP, FRAME_BUFFER_SIZE

from PIL import Image
import imageio


class Environment:

    def __init__(self, worker_index=0):

        self.env_no_frame_skip = gym.make(ENV)
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env_no_frame_skip)
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.reset()

        self.render = False
        self.images = []
        self.tmp_name = 'tmp_{}.png'.format(worker_index)

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return (84, 84, FRAME_BUFFER_SIZE)

    def get_action_size(self):
        try:
            return self.env.action_space.n
        except AttributeError:
            return self.env.action_space.shape[0]

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
        # Reset the frame buffer with FRAME_BUFFER_SIZE frames
        self.frame_buffer.append(self.env.reset())
        for _ in range(FRAME_BUFFER_SIZE-1):
            self.frame_buffer.append(self.env_no_frame_skip.step(0)[0])
        return self._convert_process_buffer()

    def act(self, action, gif=False):
        if not gif:
            return self._act(action)
        else:
            return self._act_gif(action)

    def _act(self, action):
        # Check whether the action is valid
        assert self.env.action_space.contains(action)

        if self.render:
            self.env.render()

        s, r, done, info = self.env.step(action)
        self.frame_buffer.append(s)
        return self._convert_process_buffer(), r, done, info

    def _act_gif(self, action):
        # Check whether the action is valid
        assert self.env.action_space.contains(action)

        r = 0
        i, done = 0, False
        while i < (FRAME_SKIP + 1) and not done:
            if self.render:
                self.env_no_frame_skip.render()

            # Save image
            img = Image.fromarray(self.env.render(mode='rgb_array'))
            img.save(self.tmp_name)
            self.images.append(imageio.imread(self.tmp_name))

            s_, r_tmp, done, info = self.env_no_frame_skip.step(action)
            r += r_tmp
            i += 1

        self.frame_buffer.append(s_)
        return self._convert_process_buffer(), r, done, info

    def save_gif(self, path, i):
        path = path + "_{}.gif".format(i)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

    def close(self):
        self.env.render(close=True)
        self.env.close()
