
import os
import gym

from settings import Settings

from PIL import Image
import imageio


class Environment:
    """
    Gym-environment wrapper to add the possibility to save GIFs.

    If the boolean gif if True, then every time the action methods is called,
    the environment keeps a picture of the environment in a list until the
    method save_gif is called.
    """

    def __init__(self):

        self.env = gym.make(Settings.ENV)
        self.render = False
        self.gif = False
        self.name_gif = 'save_'
        self.n_gif = {}
        self.images = []

    def set_render(self, render):
        if not render:
            self.env.render(close=True)
        self.render = render and Settings.DISPLAY

    def set_gif(self, gif, name=None):
        """
        Set the gif value and the name under which to save it.
        """
        self.gif = gif and Settings.DISPLAY
        if name is not None:
            self.name_gif = name

    def reset(self):
        if self.gif:
            self.save_gif()
        return self.env.reset()

    def act_random(self):
        return self.env.action_space.sample()

    def act(self, action):
        """
        Wrapper method to add frame skip.
        """
        r, i, done = 0, 0, False
        while i < (Settings.FRAME_SKIP + 1) and not done:
            if self.render:
                self.env.render()

            if self.gif:
                # Add image to the memory list
                img = Image.fromarray(self.env.render(mode='rgb_array'))
                img.save('tmp.png')
                self.images.append(imageio.imread('tmp.png'))

            s_, r_tmp, done, info = self.env.step(action)
            r += r_tmp
            i += 1
        return s_, r, done, info

    def save_gif(self):
        """
        If images have been saved in the memory list, save these images in a
        gif. The gif will have the name given in the set_gif method (default to
        'save_') plus a number corresponding to the number of gifs saved with
        that name plus one.

        For instance, if set_gif is called twice with name='example_gif' and
        once with name='other_example_gif', then three gifs will be saved with
        the names 'example_gif_0', 'example_gif_1' and 'other_example_gif_0'.

        The gif number wraps to 0 after Settings.MAX_NB_GIF (which will
        overwrite the first gif saved).
        """
        if not self.images:
            return

        print("Saving gif in ", Settings.GIF_PATH, "...", sep='')

        number = self.n_gif.get(self.name_gif, 0)
        path = Settings.GIF_PATH + self.name_gif + str(number) + ".gif"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []

        self.n_gif[self.name_gif] = (number + 1) % Settings.MAX_NB_GIF
        self.name_gif = 'save_'
        
        print("Gif saved!\n")

    def close(self):
        """
        Close the environment and save gif under the name 'last_gif'
        """
        self.env.render(close=True)
        if self.gif:
            self.name_gif = 'last_gif_'
            self.save_gif()
        self.env.close()
