
"""
This class is used to display in real time and save figures of the episode
rewards of an agent using matplotlib. It can also display the Q-value
distribution in real time.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from settings import Settings

plt.ion()       # Set matplotlib to interactive mode (to display in real-time)


class Displayer:

    def __init__(self):
        # List of the episode rewards
        self.rewards = []

    def add_reward(self, reward, plot=False):
        """
        Method to add an episode reward in the displayer buffer

        Args:
            reward: The reward to append to the buffer
            plot  : whether the graph must be displayed or not
        """
        self.rewards.append(reward)
        if plot:
            if Settings.DISPLAY:
                self.disp()
            else:
                print(self.rewards[-10:])
        plt.close(2)       # Close the distrib graph if it was drawn previously

    def disp(self):
        """
        Method to display the graph of the evolution of the episode rewards
        saved so far.
        """
        # Display also a smoothed curve over 10 values
        mean_reward = [np.mean(self.rewards[max(1, i - 10):i])
                       for i in range(2, len(self.rewards))]
        curves = [("results/Reward", self.rewards),
                  ("results/Mean_reward", mean_reward)]

        # The file where the graph must be saved
        fig_name = "results/Reward.png"

        if Settings.DISPLAY:
            fig = plt.figure(1)
            fig.clf()
            for path, data in curves:
                plt.plot(data)
            os.makedirs(os.path.dirname(fig_name), exist_ok=True)
            fig.savefig(fig_name)
            plt.show(block=False)
            plt.pause(0.05)      # gives plt the time to display

        # If DISPLAY is False, save the data as raw strings
        else:
            for path, data in curves:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                data = " ".join(map(str, data))
                with open(path, "w") as file:
                    file.write(data)

    def disp_distrib(self, z, delta_z, distrib, value):
        """
        Method to display the Q-value distribution over a support z
        Args:
            z      : the support of the distribution
            delta_z: the width of the atoms
            distrib: a list containing a distribution (= a list with NB_ATOMS
                        value whose sum is 1) for each possible action
            value  : a list containing the expected value of the given distrib
        """
        fig = plt.figure(2)
        fig.clf()
        for i in range(Settings.ACTION_SIZE):
            p = plt.subplot(Settings.ACTION_SIZE, 1, i+1)
            plt.bar(z, distrib[i], delta_z, label="action %i" % i)
            p.axvline(value[i], color='red', linewidth=0.7)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.05)
