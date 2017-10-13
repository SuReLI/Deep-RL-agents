
import random
import numpy as np
import tensorflow as tf
from collections import deque

from Environment import Environment
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from QNetwork import Actor, Critic

from Displayer import DISPLAYER
from Saver import SAVER
import parameters


def updateTargetGraph(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    tau = parameters.UPDATE_TARGET_RATE
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) +
            ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")
        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()
        self.low_bound, self.high_bound = self.env.get_bounds()

        print("Creation of the main Actor")
        self.mainActor = Actor(self.state_size, self.action_size,
                               self.low_bound, self.high_bound,
                               scope='mainActor')

        print("Creation of the target Actor")
        self.targetActor = Actor(self.state_size, self.action_size,
                                 self.low_bound, self.high_bound,
                                 scope='targetActor')

        print("Creation of the main Critic")
        self.mainCritic = Critic(self.state_size, self.action_size,
                                 scope='mainCritic')

        print("Creation of the target Critic")
        self.targetCritic = Critic(self.state_size, self.action_size,
                                   scope='targetCritic')

        self.buffer = PrioritizedReplayBuffer(parameters.BUFFER_SIZE,
                                              parameters.PRIOR_ALPHA)

        self.epsilon = parameters.EPSILON_START
        self.epsilon_decay = (parameters.EPSILON_START -
                              parameters.EPSILON_STOP) \
            / parameters.EPSILON_STEPS

        self.beta = parameters.PRIOR_BETA_START
        self.beta_incr = (parameters.PRIOR_BETA_STOP -
                          parameters.PRIOR_BETA_START) \
            / parameters.PRIOR_BETA_STEPS

        self.update_target_ops = updateTargetGraph(trainables)

        self.best_run = -1e10

    def run(self):

        self.total_steps = 0

        while self.total_steps < (parameters.PRE_TRAIN_STEPS +
                                  parameters.TRAINING_STEPS):

            pre_training = (self.total_steps <= parameters.PRE_TRAIN_STEPS)
            if self.total_steps == parameters.PRE_TRAIN_STEPS:
                print("End of the pre training")

            s = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            memory = deque()
            discount_R = 0

            while episode_step < parameters.MAX_EPISODE_STEPS and not done:

                if pre_training or random.random() < self.epsilon:
                    a = np.random.uniform(-self.bound, self.bound)
                    a = np.array(a)
                else:
                    a = self.sess.run(self.mainQNetwork.predict,
                                      feed_dict={
                                          self.mainQNetwork.inputs_state: [s]})
                    a = a[0]
                    a = np.clip(a, -2, 2)

                # Epsilon decay
                if self.epsilon > parameters.EPSILON_STOP:
                    self.epsilon -= self.epsilon_decay

                s_, r, done, info = self.env.act(a)

                memory.append((s, a, r, s_, done))

                if len(memory) <= parameters.N_STEP_RETURN:
                    discount_R += parameters.DISCOUNT**(len(memory) - 1) * r

                else:
                    s_mem, a_mem, r_mem, ss_mem, done_mem = memory.popleft()
                    discount_R = (discount_R - r_mem) / parameters.DISCOUNT +\
                        parameters.DISCOUNT_N * r
                    self.buffer.add(s_mem, a_mem, discount_R, ss_mem, done_mem)

                episode_reward += r
                s = s_

                episode_step += 1

                if not pre_training and \
                        episode_step % parameters.TRAINING_FREQ == 0:

                    # Beta "decay"
                    if self.beta < parameters.PRIOR_BETA_STOP:
                        self.beta += self.beta_incr
                    train_batch = self.buffer.sample(parameters.BATCH_SIZE,
                                                     self.beta)

                    feed_dict = {self.targetQNetwork.inputs_action: train_batch[1],
                                 self.targetQNetwork.inputs_state: train_batch[3]}
                    targetQvalue = self.sess.run(self.targetQNetwork.Qvalue,
                                                 feed_dict=feed_dict)

                    # Done multiplier :
                    # equals 0 if the episode was done
                    # equals 1 else
                    done_multiplier = (1 - train_batch[4])
                    targetQvalues = train_batch[2] + \
                        parameters.DISCOUNT * targetQvalue.T * done_multiplier
                    targetQvalues = targetQvalues[0]

                    feed_dict = {self.mainQNetwork.inputs_state: train_batch[0],
                                 self.mainQNetwork.inputs_action: train_batch[1],
                                 self.mainQNetwork.Qtarget: targetQvalues}
                    _, _ = self.sess.run([self.mainQNetwork.train_critic,
                                          self.mainQNetwork.train_actor],
                                         feed_dict=feed_dict)

                    update_target(self.update_target_ops, self.sess)

            if pre_training:
                self.best_run = max(self.best_run, episode_reward)

            else:
                DISPLAYER.add_reward(episode_reward)
                if episode_reward > self.best_run and \
                        self.total_steps > 1000 + parameters.PRE_TRAIN_STEPS:
                    self.best_run = episode_reward
                    print("Save best", episode_reward)
                    SAVER.save('best', self.buffer)
                    self.play_gif('results/gif/best.gif')

            self.total_steps += 1

            if self.total_steps % 500 == 0:
                print("Episode", self.total_steps, ", reward", episode_reward)

            # Save the model
            if not pre_training and self.total_steps % 5000 == 0:
                SAVER.save(self.total_steps, self.buffer)

    def play(self, number_run):
        print("Playing for", number_run, "runs")

        self.env.set_render(True)
        try:
            for _ in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    a = self.sess.run(self.mainQNetwork.predict,
                                      feed_dict={
                                          self.mainQNetwork.inputs_state: [s]})
                    a = a[0]
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

    def play_gif(self, path):

        try:
            s = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                a = self.sess.run(self.mainQNetwork.predict,
                                  feed_dict={
                                      self.mainQNetwork.inputs_state: [s]})
                a = a[0]
                s, r, done, info = self.env.act_gif(a)

                episode_reward += r
            print("Episode reward :", episode_reward)
            self.env.save_gif(path)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")
            self.env.close()

    def stop(self):
        self.env.close()
