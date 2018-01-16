
import os
import gym
from settings import ENV, FRAME_SKIP, DISPLAY, MAX_NB_GIF, GIF_PATH

from PIL import Image
import imageio


class Environment:

    def __init__(self):

        self.env_no_frame_skip = gym.make(ENV)
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env_no_frame_skip)
        print()
        self.render = False
        self.gif = False
        self.name_gif = 'save_'
        self.n_gif = {}
        self.images = []

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return list(self.env.observation_space.shape)

    def get_action_size(self):
        try:
            return self.env.action_space.n
        except AttributeError:
            return self.env.action_space.shape[0]

    def set_render(self, render):
        if not render:
            self.env.render(close=True)
        self.render = render and DISPLAY

    def set_gif(self, gif, name=None):
        self.gif = gif and DISPLAY
        if name is not None:
            self.name_gif = name

    def reset(self):
        if self.gif:
            self.save_gif()
        return self.env.reset()

    def act(self, action):
        if not self.gif:
            if self.render:
                self.env.render()
            return self.env.step(action)

        r = 0
        i, done = 0, False
        while i < (FRAME_SKIP + 1) and not done:
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

    def save_gif(self):
        if self.images == []:
            return

        print("Saving gif in ", GIF_PATH, "...", sep='')

        number = self.n_gif.get(self.name_gif, 0)
        path = GIF_PATH + self.name_gif + str(number) + ".gif"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

        self.n_gif[self.name_gif] = (number + 1) % MAX_NB_GIF
        self.name_gif = 'save_'

    def close(self):
        self.env.render(close=True)
        if self.gif:
            self.name_gif = 'last_gif_'
            self.save_gif()
        self.env.close()
