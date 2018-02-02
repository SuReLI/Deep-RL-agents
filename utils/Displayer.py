
import os

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def save(display, saver, fig_name):
    if display:
        fig = plt.figure(1)
        fig.clf()
        for path, data in saver:
            plt.plot(data)
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        fig.savefig(fig_name)
        plt.show(block=False)
        plt.pause(0.05)
    else:
        for path, data in saver:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = " ".join(map(str, data))
            with open(path, "w") as file:
                file.write(data)


class Displayer:

    def __init__(self, settings):
        self.rewards = []
        self.settings = settings

    def add_reward(self, reward, plot=False):
        self.rewards.append(reward)
        if plot:
            if self.settings.DISPLAY:
                self.disp()
            else:
                print(self.rewards[-10:])

    def disp(self):
        mean_reward = [np.mean(self.rewards[max(1, i - 10):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(self.settings.DISPLAY, saver, "results/Reward.png")
