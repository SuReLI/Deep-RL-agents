
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
        self.low_bound, self.high_bound = self.env.get_bounds()

        self.buffer = ExperienceBuffer()

        print("Creation of the actor-critic network")
        self.network = Network(self.state_size, self.action_size,
                               self.low_bound, self.high_bound)

        self.best_run = -1e10
        self.n_gif = 0

        self.sess.run(tf.global_variables_initializer())

    def run(self):

        self.total_steps = 0
        self.sess.run(self.network.target_init)

        for ep in range(1, settings.TRAINING_STEPS+1):

            episode_reward = 0
            episode_step = 0
            done = False

            # Initialize exploration noise process
            noise_process = np.zeros(self.action_size)
            noise_scale = (settings.NOISE_SCALE_INIT *
                           settings.NOISE_DECAY**ep) * \
                (self.high_bound - self.low_bound)

            # Initial state
            s = self.env.reset()
            if ep % settings.RENDER_FREQ == 0 and settings.DISPLAY:
                self.env.set_render(True)
            else:
                self.env.set_render(False)

            gif = (ep % settings.GIF_FREQ == 0) and settings.DISPLAY

            while episode_step < settings.MAX_EPISODE_STEPS and not done:

                # choose action based on deterministic policy
                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: s[None]})

                # add temporally-correlated exploration noise to action
                noise_process = settings.EXPLO_THETA * \
                    (settings.EXPLO_MU - noise_process) + \
                    settings.EXPLO_SIGMA * np.random.randn(self.action_size)

                a += noise_scale * noise_process

                s_, r, done, info = self.env.act(a, gif)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 0.0 if done else 1.0))

                # update network weights to fit a minibatch of experience
                if self.total_steps % settings.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= settings.BATCH_SIZE:

                    minibatch = self.buffer.sample()

                    q, _, _ = self.sess.run([self.network.q_values_of_given_actions, self.network.critic_train_op, self.network.actor_train_op],
                                         feed_dict={
                        self.network.state_ph: np.asarray([elem[0] for elem in minibatch]),
                        self.network.action_ph: np.asarray([elem[1] for elem in minibatch]),
                        self.network.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                        self.network.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                        self.network.is_not_done_ph: np.asarray([elem[4] for elem in minibatch])})

                    DISPLAYER.add_q(q[0])

                    # update target networks
                    _ = self.sess.run(self.network.update_targets)

                s = s_
                episode_step += 1
                self.total_steps += 1

            if gif:
                self.env.save_gif('results/gif/gif_save', self.n_gif)
                self.n_gif = (self.n_gif + 1) % 5

#            if ep > 50 and episode_reward > self.best_run:
#                print("Saving best")
#                self.play(1, 'results/gif/gif_best')
#                self.best_run = episode_reward

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
            self.env.close()

    def close(self):
        self.env.close()
