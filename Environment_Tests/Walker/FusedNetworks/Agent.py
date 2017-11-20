
import tensorflow as tf
import numpy as np

import gym
import random

from QNetwork import Network

from Environment import Environment
from ExperienceBuffer import ExperienceBuffer

from Displayer import DISPLAYER
from Saver import SAVER
import parameters


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

        self.epsilon = parameters.EPSILON_START
        self.epsilon_decay = (parameters.EPSILON_START -
                              parameters.EPSILON_STOP) \
            / parameters.EPSILON_STEPS

        self.best_run = -1e10
        self.n_gif = 0

        self.sess.run(tf.global_variables_initializer())

    def run(self):

        self.total_steps = 0

        for ep in range(1, parameters.TRAINING_STEPS + 1):

            episode_reward = 0
            episode_step = 0
            done = False

            # Initial state
            s = self.env.reset()
            self.env.set_render(ep % 1000 == 0)
            gif = (ep % 1500 == 0)
            step_allonge = ep // 1000

            while episode_step < parameters.MAX_EPISODE_STEPS + step_allonge \
                    and not done:

                if random.random() < self.epsilon:
                    a = self.env.random()
                else:
                    # choose action based on deterministic policy
                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: [s]})

                s_, r, done, info = self.env.act(a, gif)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 0.0 if done else 1.0))

                # update network weights to fit a minibatch of experience
                if self.total_steps % parameters.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= parameters.BATCH_SIZE:

                    minibatch = self.buffer.sample()

                    _, _ = self.sess.run([self.network.critic_train_op, self.network.actor_train_op],
                                         feed_dict={
                        self.network.state_ph: np.asarray([elem[0] for elem in minibatch]),
                        self.network.action_ph: np.asarray([elem[1] for elem in minibatch]),
                        self.network.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                        self.network.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                        self.network.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch])})

                    # update target networks
                    _ = self.sess.run(self.network.update_slow_targets_op)

                s = s_
                episode_step += 1
                self.total_steps += 1

            # Decay epsilon
            if self.epsilon > parameters.EPSILON_STOP:
                self.epsilon -= self.epsilon_decay

            if gif:
                self.env.save_gif('results/gif/', self.n_gif)
                self.n_gif = (self.n_gif + 1) % 5

            if episode_reward > self.best_run:
                self.best_run = episode_reward
                print("Save best", episode_reward)
                SAVER.save('best')

            DISPLAYER.add_reward(episode_reward)
            if ep % 50 == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %7.3f'
                      ' (max step: %i)' % (ep, episode_reward, episode_step,
                                           self.epsilon,
                                           parameters.MAX_EPISODE_STEPS +
                                           step_allonge))
            if ep % 500 == 0:
                DISPLAYER.disp()

    def play(self, number_run, path=''):
        print("Playing for", number_run, "runs")

        self.env.set_render(True)
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
