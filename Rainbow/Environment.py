
import os
import gym
from parameters import ENV, FRAME_SKIP

from PIL import Image
import imageio


class Environment:

    def __init__(self):

        self.env_no_frame_skip = gym.make(ENV)
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env)
        print()
        self.render = False
        self.offset = 0
        self.images = []

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return list(self.env.observation_space.shape)

    def get_action_size(self):
        if ENV == "SpaceInvaders-v0" or ENV == "SpaceInvaders-ram-v0":
            return 4
        elif ENV == "Pong-v0" or ENV == "Pong-ram-v0":
            self.offset = 2
            return 2
        else:
            try:
                return self.env.action_space.n
            except AttributeError:
                return self.env.action_space.shape[0]

    def set_render(self, render):
        self.render = render

    def reset(self):
        return self.env.reset()

    def act(self, action):
        action += self.offset
        assert self.env.action_space.contains(action)
        if self.render:
            self.env.render()
        return self.env.step(action)

    def act_gif(self, action):
        action += self.offset
        assert self.env.action_space.contains(action)
        r = 0
        i, done = 0, False
        while i < (parameters.FRAME_SKIP + 1) and not done:
            if self.render:
                self.env_no_frame_skip.render()

            #Save image
            img = Image.fromarray(self.env.render(mode='rgb_array'))
            img.save('tmp.png')
            self.images.append(imageio.imread('tmp.png'))

            s_, r_tmp, done, info = self.env_no_frame_skip.step(action)
            r += r_tmp
            i += 1
        return s_, r, done, info

    def save_gif(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

    def close(self):
        self.env.render(close=True)
        self.env.close()
