
import random
import numpy as np
import tensorflow as tf
from collections import deque

from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer
from Environment import Environment

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

        print("Creation of the main QNetwork...")
        self.mainQNetwork = QNetwork(self.sess, self.state_size,
                                     self.action_size,
                                     trainable=True, scope='main')
        print("Main QNetwork created !\n")

        print("Creation of the target QNetwork...")
        self.targetQNetwork = QNetwork(self.sess, self.state_size,
                                       self.action_size,
                                       trainable=False, scope='target')
        print("Target QNetwork created !\n")

        self.mainQNetwork.build_target_update(self.targetQNetwork.vars)

        self.buffer = ExperienceBuffer()
        self.epsilon = settings.EPSILON_START

        self.nb_ep = 1
        self.best_run = -1e10
        self.n_gif = 0

    def pre_train(self):
        print("Beginning of the pre-training...")

        for i in range(settings.PRE_TRAIN_STEPS):

            s = self.env.reset()
            done = False
            episode_step = 0
            episode_reward = 0

            while episode_step < settings.MAX_EPISODE_STEPS and not done:

                a = random.randint(0, self.action_size - 1)
                s_, r, done, info = self.env.act(a)
                self.buffer.add((s, a, r, s_, done))

                s = s_
                episode_reward += r
                episode_step += 1

            if i % (settings.PRE_TRAIN_STEPS//10) == 0:
                print("Pre-train step n", i)

            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def run(self):

        print("Beginning of the run...")

        self.pre_train()
        self.mainQNetwork.init_update_target()




        z = self.sess.run(self.mainQNetwork.z)
        import matplotlib.pyplot as plt
        plt.ion()
        self.env.set_render(True)





        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < settings.TRAINING_STEPS:

            s = self.env.reset()
            episode_reward = 0
            done = False

            episode_step = 1
            max_step = settings.MAX_EPISODE_STEPS + self.nb_ep // settings.EP_ELONGATION

            # Render settings
            self.env.set_render(self.nb_ep % settings.RENDER_FREQ == 0)
            gif = (self.nb_ep % settings.GIF_FREQ == 0) and settings.DISPLAY

            while episode_step < max_step and not done:

                if random.random() < self.epsilon:
                    a = random.randint(0, self.action_size - 1)
                else:
                    a, distr = self.sess.run([self.mainQNetwork.action, self.mainQNetwork.Q_distrib],
                                       feed_dict={self.mainQNetwork.state_ph: [s]})
                    a = a[0]

                    # print(distr[0])

                    if self.nb_ep % 250 == 0:
                        self.env.set_render(True)
                        plt.clf()
                        plt.subplot(2, 1, 1)
                        plt.bar(z, distr[0][0], label="left")
                        plt.legend()
                        plt.subplot(2, 1, 2)
                        plt.bar(z, distr[0][1], label="right")
                        plt.legend()
                        plt.show()
                        plt.pause(0.05)


                s_, r, done, info = self.env.act(a, gif)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                if episode_step % settings.TRAINING_FREQ == 0:
                    batch = self.buffer.sample()
                    self.mainQNetwork.train_minibatch(np.asarray(batch))
                    self.mainQNetwork.update_target()

                s = s_
                episode_step += 1
                self.total_steps += 1

            self.nb_ep += 1

            # Decay epsilon
            if self.epsilon > settings.EPSILON_STOP:
                self.epsilon -= settings.EPSILON_DECAY

            DISPLAYER.add_reward(episode_reward)
            # if episode_reward > self.best_run and \
            #         self.nb_ep > 50 + settings.PRE_TRAIN_STEPS:
            #     self.best_run = episode_reward
            #     print("Save best", episode_reward)
            #     SAVER.save('best')
            #     self.play(1, 'results/gif/best.gif')

            # if gif:
            #     self.env.save_gif('results/gif/gif_save', self.n_gif)
            #     self.n_gif = (self.n_gif + 1) % settings.MAX_NB_GIF

            self.total_steps += 1

            if self.nb_ep % settings.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %f'
                      ', Max steps: %i' % (self.nb_ep, episode_reward,
                                           episode_step, self.epsilon,
                                           max_step))

            # Save the model
            if self.nb_ep % settings.SAVE_FREQ == 0:
                SAVER.save(self.nb_ep)

    def play(self, number_run, path=''):
        print("Playing for", number_run, "runs")

        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    a, = self.sess.run(self.mainQNetwork.action,
                                      feed_dict={self.mainQNetwork.state_ph: [s]})
                    a = a[0]
                    s, r, done, info = self.env.act(a, path != '')


                    episode_reward += r

                print("Episode reward :", episode_reward)

                if path != '':
                    self.env.save_gif(path, i)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            self.env.set_render(False)
            print("End of the demo")
            self.env.close()

    def stop(self):
        self.env.close()
