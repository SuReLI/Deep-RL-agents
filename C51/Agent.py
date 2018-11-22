
import signal
import threading

import tensorflow as tf
import numpy as np
import random
from collections import deque

from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer
from Environment import Environment

from settings import Settings


class Agent:
    """
    This class builds an agent with its own QNetwork, memory buffer and
    environment to learn a policy.
    """

    def __init__(self, sess, gui, displayer, saver):
        """
        Build a new instance of Environment, QNetwork and ExperienceBuffer.

        Args:
            sess     : the tensorflow session in which to build the network
            gui      : a GUI instance to manage the control of the agent
            displayer: a Displayer instance to keep track of the episode rewards
            saver    : a Saver instance to save periodically the network
        """
        print("Initializing the agent...")

        self.sess = sess
        self.gui = gui
        self.gui_thread = threading.Thread(target=self.gui.run)
        self.displayer = displayer
        self.saver = saver
        signal.signal(signal.SIGINT, self.interrupt)

        self.env = Environment()
        self.QNetwork = QNetwork(self.sess)
        self.buffer = ExperienceBuffer()
        self.epsilon = Settings.EPSILON_START

        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NB_ATOMS - 1)
        self.z = np.linspace(Settings.MIN_Q, Settings.MAX_Q, Settings.NB_ATOMS)

        self.create_summaries()

        self.best_run = -1e10
        self.n_gif = 0

        print("Agent initialized !\n")

    def create_summaries(self):

        self.ep_reward_ph = tf.placeholder(tf.float32)
        ep_reward_summary = tf.summary.scalar("Episode/Episode reward", self.ep_reward_ph)

        self.steps_ph = tf.placeholder(tf.float32)
        steps_summary = tf.summary.scalar("Episode/Nb steps", self.steps_ph)

        self.epsilon_ph = tf.placeholder(tf.float32)
        epsilon_summary = tf.summary.scalar("Settings/Epsilon", self.epsilon_ph)

        self.ep_summary = tf.summary.merge([ep_reward_summary,
                                            epsilon_summary,
                                            steps_summary])

        self.lr_ph = tf.placeholder(tf.float32)
        self.lr_summary = tf.summary.scalar("Settings/Learning rate", self.lr_ph)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def pre_train(self):
        """
        Method to run a random agent in the environment to fill the memory
        buffer.
        """
        print("Beginning of the pre-training...")

        for i in range(Settings.PRE_TRAIN_EPS):

            s = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while episode_step < Settings.MAX_EPISODE_STEPS and not done:

                a = self.env.act_random()
                s_, r, done, info = self.env.act(a)
                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                s = s_
                episode_reward += r
                episode_step += 1

            if Settings.PRE_TRAIN_EPS > 5 and i % (Settings.PRE_TRAIN_EPS // 5) == 0:
                print("Pre-train step n", i)

            # Set the best score to at least the max score the random agent got
            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def save_best(self, episode_reward):
        self.best_run = episode_reward
        print("Save best", episode_reward)
        self.saver.save('best')
        # self.play(1, 'best')

    def run(self):
        """
        Method to run the agent in the environment to collect experiences and
        learn on these experiences by gradient descent.
        """
        print("Beginning of the run...")

        self.pre_train()
        self.QNetwork.init_target()
        self.gui_thread.start()

        self.nb_ep = 1
        learning_steps = 0

        while self.nb_ep < Settings.TRAINING_EPS and not self.gui.STOP:

            s = self.env.reset()
            episode_reward = 0
            done = False

            episode_step = 1
            # The more episodes the agent performs, the longer they are
            max_step = Settings.MAX_EPISODE_STEPS
            if Settings.EP_ELONGATION > 0:
                max_step += self.nb_ep // Settings.EP_ELONGATION

            # Render settings
            self.env.set_render(self.gui.render.get(self.nb_ep))
            self.env.set_gif(self.gui.gif.get(self.nb_ep))
            plot_distrib = self.gui.plot_distrib.get(self.nb_ep)

            while episode_step <= max_step and not done:

                # Exploration by epsilon-greedy policy
                if random.random() < self.epsilon:
                    a = self.env.act_random()
                else:
                    Qdistrib = self.QNetwork.act(s)
                    Qvalue = np.sum(self.z * Qdistrib, axis=1)
                    a = np.argmax(Qvalue, axis=0)

                    if plot_distrib:
                        self.displayer.disp_distrib(self.z, self.delta_z,
                                                    Qdistrib, Qvalue)

                s_, r, done, info = self.env.act(a)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                if episode_step % Settings.TRAINING_FREQ == 0:
                    batch = self.buffer.sample()
                    self.QNetwork.train(np.asarray(batch))
                    self.QNetwork.update_target()

                    feed_dict = {self.lr_ph: self.QNetwork.learning_rate}
                    summary = self.sess.run(self.lr_summary, feed_dict=feed_dict)
                    self.writer.add_summary(summary, learning_steps)
                    learning_steps += 1

                s = s_
                episode_step += 1

            # Decay epsilon
            if self.epsilon > Settings.EPSILON_STOP:
                self.epsilon -= Settings.EPSILON_DECAY

            self.displayer.add_reward(episode_reward, plot=self.gui.plot.get(self.nb_ep))
            # if episode_reward > self.best_run:
            #     self.save_best(episode_reward)
            
            # Episode display
            if self.gui.ep_reward.get(self.nb_ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %f, Max steps: %i, LR: %fe-4' % (
                    self.nb_ep, episode_reward, episode_step, self.epsilon, max_step, self.QNetwork.learning_rate))

            # Write the summary
            feed_dict = {self.ep_reward_ph: episode_reward,
                         self.epsilon_ph: self.epsilon,
                         self.steps_ph: episode_step}
            summary = self.sess.run(self.ep_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.nb_ep)

            # Save the model
            if self.gui.save.get(self.nb_ep):
                self.saver.save(self.nb_ep)

            self.nb_ep += 1

        print("Training completed !")
        self.env.close()
        self.display()
        self.gui_thread.join()

    def play(self, number_run, name=None):
        """
        Method to evaluate the policy without exploration.

        Args:
            number_run: the number of episodes to perform
            name      : the name of the gif that will be saved
        """
        print("Playing for", number_run, "runs")

        self.env.set_render(Settings.DISPLAY)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False
                self.env.set_gif(True, name)

                while not done:
                    Qdistrib = self.QNetwork.act(s)
                    Qvalue = np.sum(self.z * Qdistrib, axis=1)
                    a = np.argmax(Qvalue, axis=0)
                    s, r, done, info = self.env.act(a)

                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")

    def display(self):
        self.displayer.disp()

    def stop(self):
        self.env.close()

    def interrupt(self, sig, frame):
        self.gui.stop_run()
