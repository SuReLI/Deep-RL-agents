
import os

import numpy as np
import matplotlib.pyplot as plt

import settings

plt.ion()

def save(saver, fig_name):
    if settings.DISPLAY:
        for path, data in saver:
            plt.plot(data)
        fig = plt.gcf()
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        fig.savefig(fig_name)
        plt.show(block=False)
        plt.pause(0.05)
        fig.clf()
    else:
        for path, data in saver:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = " ".join(map(str, data))
            with open(path, "w") as file:
                file.write(data)


class Displayer:

    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) % settings.PLOT_FREQ == 0:
            if settings.DISPLAY:
                self.disp()
            else:
                print(self.rewards[-50:])

    def disp(self):
        mean_reward = [np.mean(self.rewards[max(1, i - 100):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(saver, "results/Reward.png")

DISPLAYER = Displayer()
