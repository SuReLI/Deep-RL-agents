
import os

import numpy as np
import matplotlib

from settings import Settings

if Settings.DISPLAY:
    import matplotlib.pyplot as plt
    plt.ion()     # Set matplotlib to interactive mode (to display in real-time)

else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


class Displayer:
    """
    This class is used to display in real time and save figures of the episode
    rewards of an agent or many agents using matplotlib. It can also display the
    Q-value distribution in real time.
    """

    def __init__(self):
        # List of the episode rewards
        if hasattr(Settings, 'NB_ACTORS'):
            nb_actors = Settings.NB_ACTORS
        else:
            nb_actors = 1

        self.rewards = [[] for i in range(nb_actors)]

    def mean(self, tab, step=10):
        """
        Return a smoothed version of a graph over 'step' steps.
        """
        return [np.mean(tab[max(1, i - step):i]) for i in range(2, len(tab))]

    def save(self, curves, fig_name):
        """
        Plot a list of curves in a figure and save this figure under the name
        'fig_name'.

        Args:
            curves  : a list of tuples (path, data) with
                        data : a list with the data to plot
                        path : a path where to save the data if the plotting is
                                not possible
            fig_name: the name of the figure to be saved
        """
        fig = plt.figure(1)
        fig.clf()
        for path, data in curves:
            plt.plot(data)
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        fig.savefig(fig_name)
        if Settings.DISPLAY:
            plt.show(block=False)
            plt.pause(0.05)      # gives plt the time to display

    def add_reward(self, reward, n_agent=0, plot=False):
        """
        Method to add an episode reward in the displayer buffer

        Args:
            reward : the reward to append to the buffer
            n_agent: the number of the agent that collected this reward
            plot   : whether the graph must be displayed or not
        """
        self.rewards[n_agent].append(reward)
        if plot:
            self.disp()
        plt.close(2)       # Close the distrib graph if it was drawn previously

    def disp(self, mode='one'):
        """
        Method to display the graph of episode rewards saved so far.
        """
        if mode not in ('one', 'seq', 'all'):
            print("Error : this display mode is not supported.")
            return

        if mode == 'one':
            curves = [(Settings.RESULTS_PATH + "Reward", self.rewards[0]),
                      (Settings.RESULTS_PATH + "Mean_reward", self.mean(self.rewards[0]))]
            fig_name = Settings.RESULTS_PATH + "Reward.png"

        elif mode == 'seq':
            curves = [(Settings.RESULTS_PATH + "Seq_reward", self.seq_rewards),
                      (Settings.RESULTS_PATH + "Mean_seq_reward", self.mean(self.seq_rewards))]
            fig_name = Settings.RESULTS_PATH + "Seq_reward.png"

        elif mode == 'all':
            curves = [(Settings.RESULTS_PATH + "All_rewards_" + str(i), self.rewards[i])
                      for i in range(len(self.rewards))]
            fig_name = Settings.RESULTS_PATH + "All_reward.png"

        self.save(curves, fig_name)

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
            p = plt.subplot(Settings.ACTION_SIZE, 1, i + 1)
            plt.bar(z, distrib[i], delta_z, label="action %i" % i)
            p.axvline(value[i], color='red', linewidth=0.7)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.05)
