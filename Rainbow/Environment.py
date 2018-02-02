
import os
import gym

from PIL import Image
import imageio


class Environment:

    def __init__(self, settings):

        self.settings = settings
        self.env_no_frame_skip = gym.make(self.settings.ENV)
        self.env = gym.wrappers.SkipWrapper(self.settings.FRAME_SKIP)(self.env_no_frame_skip)
        print()
        self.render = False
        self.gif = False
        self.name_gif = 'save_'
        self.n_gif = {}
        self.images = []

    def set_render(self, render):
        if not render:
            self.env.render(close=True)
        self.render = render and self.settings.DISPLAY

    def set_gif(self, gif, name=None):
        self.gif = gif and self.settings.DISPLAY
        if name is not None:
            self.name_gif = name

    def reset(self):
        if self.gif:
            self.save_gif()
        return self.env.reset()

    def act_random(self):
        return self.env.action_space.sample()

    def act(self, action):
        if not self.gif:
            if self.render:
                self.env.render()
            return self.env.step(action)

        r = 0
        i, done = 0, False
        while i < (self.settings.FRAME_SKIP + 1) and not done:
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
        if not self.images:
            return

        print("Saving gif in ", self.settings.GIF_PATH, "...", sep='')

        number = self.n_gif.get(self.name_gif, 0)
        path = self.settings.GIF_PATH + self.name_gif + str(number) + ".gif"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

        self.n_gif[self.name_gif] = (number + 1) % self.settings.MAX_NB_GIF
        self.name_gif = 'save_'

    def close(self):
        self.env.render(close=True)
        if self.gif:
            self.name_gif = 'last_gif_'
            self.save_gif()
        self.env.close()
