
import numpy as np
import matplotlib.pyplot as plt

import parameters


def save(saver, fig_name):
    for path, data in saver:
        plt.plot(data)
    plt.show(block=False)


class Displayer:

    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) % parameters.PLOT_FREQ == 0:
            self.disp()

    def disp(self):
        mean_reward = [np.mean(self.rewards[max(1, i - 50):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(saver, "results/Reward.png")

    def reset(self):
        self.rewards = []

DISPLAYER = Displayer()
