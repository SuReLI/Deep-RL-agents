
import os

import numpy as np
import matplotlib.pyplot as plt

import parameters


def save(saver, fig_name):
    if parameters.DISPLAY:
        for path, data in saver:
            plt.plot(data)
        fig = plt.gcf()
        try:
            os.makedirs(os.path.dirname(fig_name))
        except OSError:
            pass
        fig.savefig(fig_name)
        plt.show(block=False)
    else:
        for path, data in saver:
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass
            data = " ".join(map(str, data))
            with open(path, "w") as file:
                file.write(data)


class Displayer:

    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) % parameters.PLOT_FREQ == 0:
            if parameters.DISPLAY:
                self.disp()
            else:
                print(self.rewards[-50:])

    def disp(self):
        mean_reward = [np.mean(self.rewards[max(1, i - 50):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(saver, "results/Reward.png")

DISPLAYER = Displayer()
