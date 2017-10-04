
import numpy as np
import matplotlib.pyplot as plt

import parameters


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
            plt.plot(reward)
        plt.show(block=False)

    def disp_one(self):
        reward = self.rewards[1]
        mean_reward = [np.mean(reward[max(1, i - 100):i])
                       for i in range(2, len(reward))]
        print(len(reward))
        print(len(mean_reward))
        plt.plot(reward)
        plt.plot(mean_reward)
        plt.show(block=False)

    def disp_seq(self):
        mean_reward = [np.mean(self.sequential_rewards[max(1, i - 100):i])
                       for i in range(2, len(self.sequential_rewards))]
        plt.plot(self.sequential_rewards)
        plt.plot(mean_reward)
        plt.show(block=False)


DISPLAYER = Displayer()
