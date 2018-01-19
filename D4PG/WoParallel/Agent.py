
import tensorflow as tf
import numpy as np

import gym
import random

from QNetwork import Network

from Environment import Environment
from ExperienceBuffer import BUFFER

from Displayer import DISPLAYER

import GUI
import settings


class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")

        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()[0]
        self.action_size = self.env.get_action_size()
        self.bounds = self.env.get_bounds()

        print("Creation of the actor-critic network")
        self.network = Network(self.sess,
                               self.state_size, self.action_size,
                               self.bounds)

        self.sess.run(tf.global_variables_initializer())

    def predict_action(self, s):
        return self.sess.run(self.network.actions,
                             feed_dict={self.network.state_ph: s[None]})[0]

    def run(self):

        self.total_steps = 1
        self.sess.run(self.network.target_init)        

        ep = 1
        while ep < settings.TRAINING_EPS + 1 and not GUI.STOP:

            episode_reward = 0
            episode_step = 0
            done = False

            # Initialize exploration noise process
            noise_process = np.zeros(self.action_size)
            noise_scale = settings.NOISE_SCALE * settings.NOISE_DECAY**ep

            # Initial state
            s = self.env.reset()
            self.env.set_render(GUI.render.get(ep))

            while episode_step < settings.MAX_EPISODE_STEPS and not done:

                noise = np.random.normal(size=self.action_size)
                scaled_noise = noise_scale * noise

                a = np.clip(self.predict_action(s) +
                            scaled_noise, *self.bounds)

                s_, r, done, info = self.env.act(a)

                episode_reward += r

                BUFFER.add(s, a, r, s_, 0.0 if done else 1.0)

                if self.total_steps % settings.TRAINING_FREQ == 0:
                    self.network.train(BUFFER.sample())

                s = s_
                episode_step += 1
                self.total_steps += 1

            if GUI.ep_reward.get(ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (ep, episode_reward, episode_step, noise_scale))

            plot = GUI.plot.get(ep)
            DISPLAYER.add_reward(episode_reward, plot)
            ep += 1

    def play(self, number_run):
        print("Playing for", number_run, "runs")

        self.env.set_render(settings.DISPLAY)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:

                    a = self.predict_action(s)

                    s, r, done, info = self.env.act(a)

                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            self.env.set_render(False)
            print("End of the demo")
            self.env.close()

    def close(self):
        self.env.close()
