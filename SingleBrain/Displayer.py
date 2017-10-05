
import numpy as np
import matplotlib.pyplot as plt

import parameters

def save(path, data):
    if parameters.DISPLAY:
        plt.plot(data)
    else:
        data = " ".join(map(str, data))
        with open(path, "w") as file:
            file.write(data)


class Displayer:

    def __init__(self):
        self.rewards = [[] for a in range(parameters.THREADS + 1)]
        self.sequential_rewards = []

    def add_reward(self, reward, n_agent):
        self.rewards[n_agent].append(reward)
        if n_agent != 0:
            self.sequential_rewards.append(reward)

    def disp_all(self):
        for reward in self.rewards:
            save("results/All_rewards", reward)
        if parameters.DISPLAY:
            plt.show(block=False)

    def disp_one(self):
        reward = self.rewards[1]
        mean_reward = [np.mean(reward[max(1, i - 100):i])
                       for i in range(2, len(reward))]
        save("results/One_reward", reward)
        save("results/One_mean_reward", mean_reward)
        if parameters.DISPLAY:
            plt.show(block=False)

    def disp_seq(self):
        mean_reward = [np.mean(self.sequential_rewards[max(1, i - 100):i])
                       for i in range(2, len(self.sequential_rewards))]
        save("results/Seq_reward", self.sequential_rewards)
        save("results/Seq_mean_reward", mean_reward)
        if parameters.DISPLAY:
            plt.show(block=False)


DISPLAYER = Displayer()
