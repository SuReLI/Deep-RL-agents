
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


def mean(tab, step=10):
    return [np.mean(tab[max(1, i - step):i]) for i in range(2, len(tab))]


class Displayer:

    def __init__(self):
        self.rewards = [[] for a in range(settings.NB_ACTORS + 1)]
        self.sequential_rewards = []
        self.q_buf = []

    def add_reward(self, reward, n_agent, plot=False):
        self.rewards[n_agent].append(reward)
        if n_agent != 0:
            self.sequential_rewards.append(reward)
        if plot:
            if settings.DISPLAY:
                self.disp_one()
            else:
                print(self.rewards[1][max(0, -50):])

    def disp_all(self):
        saver = [("results/All_rewards/All_rewards_" + str(i), self.rewards[i])
                 for i in range(len(self.rewards))]
        save(saver, "results/All_rewards.png")

    def disp_one(self):
        reward = self.rewards[1]
        mean_reward = mean(reward, 25)
        saver = [("results/One_reward", reward),
                 ("results/One_mean_reward", mean_reward)]
        save(saver, "results/One_reward.png")

    def disp_seq(self):
        mean_reward = mean(self.sequential_rewards)
        saver = [("results/Seq_reward", self.sequential_rewards),
                 ("results/Seq_mean_reward", mean_reward)]
        save(saver, "results/Seq_reward.png")

    def disp(self):
        self.disp_all()
        self.disp_seq()
        self.disp_one()

    def add_q(self, q):
        self.q_buf.append(q)

    def disp_q(self):
        mean_q = mean(self.q_buf)
        saver = [("results/Q", self.q_buf),
                 ("results/Q_mean", mean_q)]
        save(saver, "results/Q.png")


DISPLAYER = Displayer()
