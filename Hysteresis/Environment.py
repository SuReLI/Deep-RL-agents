
from HystEnv import HystEnv
from parameters import DISPLAY

class Environment:

    def __init__(self):

        self.env = HystEnv()
        self.render = False
        self.images = []

    def get_state_size(self):
        return [2]

    def get_action_size(self):
        return 3

    def set_render(self, render):
        self.render = render and DISPLAY

    def reset(self):
        return self.env.reset()

    def act(self, action):
        if self.render:
            self.env.render()
        return self.env.step(action)

    def close(self):
        self.env.close()
