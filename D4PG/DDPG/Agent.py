
import tensorflow as tf
import numpy as np

import gym
import random

from QNetwork import Network

from Environment import Environment
from ExperienceBuffer import ExperienceBuffer

from Displayer import DISPLAYER
from Saver import SAVER
import settings


class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")

        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()[0]
        self.action_size = self.env.get_action_size()
        self.bounds = self.env.get_bounds()
        self.low_bound, self.high_bound = self.bounds

        self.buffer = ExperienceBuffer()

        print("Creation of the actor-critic network")
        self.network = Network(self.sess, self.state_size,
                               self.action_size, self.bounds)

        self.best_run = -1e10
        self.n_gif = 0

        self.sess.run(tf.global_variables_initializer())

    def run(self):

        self.total_steps = 0
        self.sess.run(self.network.target_init)

        for ep in range(1, settings.TRAINING_STEPS + 1):

            episode_reward = 0
            episode_step = 0
            done = False

            noise_scale = settings.NOISE_SCALE * settings.NOISE_DECAY**ep

            # Initial state
            s = self.env.reset()

            render = ep % settings.RENDER_FREQ == 0 and settings.DISPLAY
            self.env.set_render(render)
            gif = (ep % settings.GIF_FREQ == 0) and settings.DISPLAY

            while episode_step < settings.MAX_EPISODE_STEPS and not done:

                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: s[None]})

                noise = np.random.normal(size=self.action_size)
                a += noise_scale * noise

                s_, r, done, info = self.env.act(a, gif)

                episode_reward += r

                self.buffer.add((s, a, r, s_, 0.0 if done else 1.0))

                if self.total_steps % settings.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= settings.BATCH_SIZE:
                    minibatch = self.buffer.sample()
                    self.network.train(minibatch)

                if (self.total_steps+1) % (500*settings.TRAINING_FREQ) == 0:
                    self.buffer.stats()

                s = s_
                episode_step += 1
                self.total_steps += 1

            if gif:
                self.env.save_gif('results/gif/gif_save', self.n_gif)
                self.n_gif = (self.n_gif + 1) % 5

            if ep % settings.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (ep, episode_reward, episode_step, noise_scale))
            DISPLAYER.add_reward(episode_reward)

    def play(self, number_run, path=''):
        print("Playing for", number_run, "runs")

        self.env.set_render(path != '')
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:

                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: s[None]})

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
            self.close()

    def close(self):
        self.env.close()
