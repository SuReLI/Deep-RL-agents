
import os

import numpy as np
import matplotlib.pyplot as plt

import settings


plt.ion()

def save(saver, fig_name):
    if settings.DISPLAY:
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

    def __init__(self):
        self.rewards = []
        self.q_buf = []

    def add_reward(self, reward, plot=False):
        self.rewards.append(reward)
        if plot:
            if settings.DISPLAY:
                self.disp()
            else:
                print(self.rewards[-50:])

    def disp(self):
        mean_reward = [np.mean(self.rewards[max(1, i - 50):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(saver, "results/Reward.png")

    def add_q(self, q):
        self.q_buf.append(q)

    def disp_q(self):
        mean_q = [np.mean(self.q_buf[max(1, i - 10):i])
                  for i in range(1, len(self.q_buf))]
        saver = [("results/Q", self.q_buf),
                 ("results/Q_mean", mean_q)]
        save(saver, "results/Q.png")


DISPLAYER = Displayer()
