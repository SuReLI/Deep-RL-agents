
import os
import gym
from settings import ENV, DISPLAY, MAX_NB_GIF, GIF_PATH

from PIL import Image
import imageio

class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        print()
        self.render = False
        self.gif = False
        self.name_gif = 'save_'
        self.n_gif = {}
        self.images = []

    def get_state_size(self):
        return list(self.env.observation_space.shape)

    def get_action_size(self):
        return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def set_render(self, render):
        if not render:
            self.env.close()
        self.render = render

    def set_gif(self, gif, name=None):
        self.gif = gif and DISPLAY
        if name is not None:
            self.name_gif = name

    def reset(self):
        if self.gif:
            self.save_gif()
        return self.env.reset()

    def random(self):
        return self.env.action_space.sample()

    def act(self, action):
        if self.gif:
            #Save image
            img = Image.fromarray(self.env.render(mode='rgb_array'))
            img.save('tmp.png')
            self.images.append(imageio.imread('tmp.png'))


        if self.render:
            self.env.render()
        return self.env.step(action)

    def save_gif(self):
        if not self.images:
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
        if self.gif:
            self.name_gif = 'last_gif_'
            self.save_gif()
        self.env.close()
