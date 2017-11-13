
import tensorflow as tf
import numpy as np

import gym
import random
from collections import deque

from QNetwork import Network

from Environment import Environment
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer

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

        self.buffer = PrioritizedReplayBuffer(parameters.BUFFER_SIZE,
                                              parameters.ALPHA)

        print("Creation of the actor-critic network...")
        self.network = Network(self.state_size, self.action_size,
                               self.low_bound, self.high_bound)
        print("Network created !\n")

        self.epsilon = parameters.EPSILON_START
        self.beta = parameters.BETA_START

        self.best_run = -1e10

        self.sess.run(tf.global_variables_initializer())

    def run(self):

        self.nb_ep = 1
        self.total_steps = 0

        for self.nb_ep in range(1, parameters.TRAINING_STEPS + 1):

            episode_reward = 0
            episode_step = 0
            done = False
            memory = deque()

            # Initial state
            s = self.env.reset()
            max_steps = parameters.MAX_EPISODE_STEPS + self.nb_ep // parameters.EP_ELONGATION

            while episode_step < max_steps and not done:

                if random.random() < self.epsilon:
                    a = self.env.random()
                else:
                    # choose action based on deterministic policy
                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: [s]})

                # Decay epsilon
                if self.epsilon > parameters.EPSILON_STOP:
                    self.epsilon -= parameters.EPSILON_DECAY

                s_, r, done, info = self.env.act(a)
                memory.append((s, a, r, s_, 0.0 if done else 1.0))

                if len(memory) > parameters.N_STEP_RETURN:
                    s_mem, a_mem, r_mem, ss_mem, done_mem = memory.popleft()
                    discount_R = 0
                    for i, (si, ai, ri, s_i, di) in enumerate(memory):
                        discount_R += ri * parameters.DISCOUNT ** (i+1)
                    self.buffer.add(s_mem, a_mem, discount_R, s_, done)

                # update network weights to fit a minibatch of experience
                if self.total_steps % parameters.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= parameters.BATCH_SIZE:

                    minibatch = self.buffer.sample(parameters.BATCH_SIZE,
                                                   self.beta)

                    if self.beta <= parameters.BETA_STOP:
                        self.beta += parameters.BETA_INCR

                    td_errors, _, _ = self.sess.run(
                        [self.network.td_errors, self.network.critic_train_op,
                            self.network.actor_train_op],
                        feed_dict={
                            self.network.state_ph: minibatch[0],
                            self.network.action_ph: minibatch[1],
                            self.network.reward_ph: minibatch[2],
                            self.network.next_state_ph: minibatch[3],
                            self.network.is_not_terminal_ph: minibatch[4]})

                    self.buffer.update_priorities(minibatch[6], td_errors+1e-6)
                    # update target networks
                    _ = self.sess.run(self.network.update_slow_targets_op)

                episode_reward += r
                s = s_
                episode_step += 1
                self.total_steps += 1

            self.nb_ep += 1

            if self.nb_ep % parameters.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon : %7.3f, Max steps : %i' %
                      (self.nb_ep, episode_reward, episode_step, self.epsilon, max_steps))

            DISPLAYER.add_reward(episode_reward)

            if episode_reward > self.best_run and self.nb_ep > 100:
                self.best_run = episode_reward
                print("Best agent ! ", episode_reward)
                SAVER.save('best')

            if self.nb_ep % parameters.SAVE_FREQ == 0:
                SAVER.save(self.nb_ep)

    def play(self, number_run):
        print("Playing for", number_run, "runs")

        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:

                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: [s]})

                    s_, r, done, info = self.env.act(a)
                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")
            self.env.close()

    def close(self):
        self.env.close()
