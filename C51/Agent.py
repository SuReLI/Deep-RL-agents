
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer
from Environment import Environment

import GUI
from Displayer import DISPLAYER
from Saver import SAVER
import settings


class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")

        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()

        print("Creation of the QNetwork...")
        self.QNetwork = QNetwork(self.sess, self.state_size,
                                 self.action_size)
        print("QNetwork created !\n")

        self.buffer = ExperienceBuffer()
        self.epsilon = settings.EPSILON_START

        self.nb_ep = 1
        self.best_run = -1e10
        self.n_gif = 0

    def print_distrib(self, distrib, value):
        fig = plt.figure(2)
        fig.clf()
        for i in range(self.action_size):
            p = plt.subplot(self.action_size, 1, i+1)
            plt.bar(self.z, distrib[i], self.delta_z, label="left")
            p.axvline(value[i], color='red', linewidth=0.7)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.05)

    def pre_train(self):
        print("Beginning of the pre-training...")

        for i in range(settings.PRE_TRAIN_STEPS):

            s = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while episode_step < settings.MAX_EPISODE_STEPS and not done:

                a = random.randint(0, self.action_size - 1)
                s_, r, done, info = self.env.act(a)
                self.buffer.add((s, a, r, s_, done))

                s = s_
                episode_reward += r
                episode_step += 1

            if settings.PRE_TRAIN_STEPS > 5 and i % (settings.PRE_TRAIN_STEPS // 5) == 0:
                print("Pre-train step n", i)

            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def run(self):

        print("Beginning of the run...")

        self.pre_train()
        self.QNetwork.init_update_target()

        self.delta_z = self.QNetwork.delta_z
        self.z = self.sess.run(self.QNetwork.z)

        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < settings.TRAINING_STEPS and not GUI.STOP:

            s = self.env.reset()
            episode_reward = 0
            done = False

            episode_step = 1
            max_step = settings.MAX_EPISODE_STEPS
            if settings.EP_ELONGATION > 0:
                max_step += self.nb_ep // settings.EP_ELONGATION

            # Render settings
            self.env.set_render(GUI.render.get(self.nb_ep))
            self.env.set_gif(GUI.gif.get(self.nb_ep))
            plot_distrib = GUI.plot_distrib.get(self.nb_ep)

            while episode_step <= max_step and not done:

                if random.random() < self.epsilon:
                    a = random.randint(0, self.action_size - 1)
                else:
                    if plot_distrib:
                        a, distr, value = self.sess.run([self.QNetwork.action, self.QNetwork.Q_distrib, self.QNetwork.Q_value],
                                                        feed_dict={self.QNetwork.state_ph: [s]})
                        a, distr, value = a[0], distr[0], value[0]
                        self.print_distrib(distr, value)

                    else:
                        a, = self.sess.run(self.QNetwork.action,
                                           feed_dict={self.QNetwork.state_ph: [s]})


                s_, r, done, info = self.env.act(a)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                if episode_step % settings.TRAINING_FREQ == 0:
                    batch = self.buffer.sample()
                    self.QNetwork.train_minibatch(np.asarray(batch))
                    self.QNetwork.update_target()

                s = s_
                episode_step += 1
                self.total_steps += 1

            self.nb_ep += 1

            # Decay epsilon
            if self.epsilon > settings.EPSILON_STOP:
                self.epsilon -= settings.EPSILON_DECAY

            DISPLAYER.add_reward(episode_reward, GUI.plot.get(self.nb_ep))
            # if episode_reward > self.best_run and \
            #         self.nb_ep > 50 + settings.PRE_TRAIN_STEPS:
            #     self.best_run = episode_reward
            #     print("Save best", episode_reward)
            #     SAVER.save('best')
            #     self.play(1, 'results/gif/best.gif')

            # Episode display setting
            if GUI.ep_reward.get(self.nb_ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %f, Max steps: %i' % (
                    self.nb_ep, episode_reward, episode_step, self.epsilon, max_step))

            # Save the model
            if GUI.save.get(self.nb_ep):
                SAVER.save(self.nb_ep)

        self.env.close()

    def play(self, number_run, path=''):
        print("Playing for", number_run, "runs")

        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False
                self.env.set_gif(True, path != '')

                while not done:
                    a, = self.sess.run(self.QNetwork.action,
                                       feed_dict={self.QNetwork.state_ph: [s]})
                    s, r, done, info = self.env.act(a)

                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")
            self.env.close()

    def stop(self):
        self.env.close()
