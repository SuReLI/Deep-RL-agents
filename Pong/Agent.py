
import random
import numpy as np
import tensorflow as tf

from Environment import Environment
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from QNetwork import QNetwork

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

        print("Creation of the main QNetwork")
        self.mainQNetwork = QNetwork(self.state_size, self.action_size, 'main')
        print("Creation of the target QNetwork")
        self.targetQNetwork = QNetwork(self.state_size, self.action_size,
                                       'target')

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

        trainables = tf.trainable_variables()
        self.update_target_ops = updateTargetGraph(trainables)

        self.best_run = -1e10

    def run(self):

        self.total_steps = 0
        self.action_mem = 0
        self.nb_same_action = 0
        self.n_gif = 0
        while self.total_steps < (parameters.PRE_TRAIN_STEPS +
                                  parameters.TRAINING_STEPS):

            pre_training = (self.total_steps <= parameters.PRE_TRAIN_STEPS)
            if self.total_steps == parameters.PRE_TRAIN_STEPS:
                print("End of the pre training")

            s = self.env.reset()
            print(s)
            episode_reward = 0
            done = False

            episode_step = 0
            gif = (self.total_steps % 200 == 0)
            self.env.set_render(gif)

            while episode_step < parameters.MAX_EPISODE_STEPS and not done:

                if pre_training or random.random() < self.epsilon:
                    a = random.randint(0, self.action_size - 1)
                else:
                    a = self.sess.run(self.mainQNetwork.predict,
                                      feed_dict={self.mainQNetwork.inputs: [s]})
                    a = a[0]

                if a == self.action_mem:
                    self.nb_same_action += 1
                else:
                    self.nb_same_action = 0
                    self.action_mem = a

                if self.nb_same_action >= 20:
                    print("Same action :", a)

                # Decay epsilon
                if self.epsilon > parameters.EPSILON_STOP:
                    self.epsilon -= self.epsilon_decay

                s_, r, done, info = self.env.act(a, gif=gif)

                self.buffer.add(s, a, r, s_, done)

                episode_reward += r
                s = s_
                episode_step += 1

                if not pre_training and \
                        episode_step % parameters.TRAINING_FREQ == 0:

                    train_batch = self.buffer.sample(parameters.BATCH_SIZE,
                                                     self.beta)
                    # Decay beta
                    if self.beta < parameters.PRIOR_BETA_STOP:
                        self.beta += self.beta_incr

                    feed_dict = {self.mainQNetwork.inputs: train_batch[3]}
                    mainQaction = self.sess.run(self.mainQNetwork.predict,
                                                feed_dict=feed_dict)

                    feed_dict = {self.targetQNetwork.inputs: train_batch[3]}
                    targetQvalues = self.sess.run(self.targetQNetwork.Qvalues,
                                                  feed_dict=feed_dict)

                    # Done multiplier :
                    # equals 0 if the episode was done
                    # equals 1 else
                    done_multiplier = (1 - train_batch[4])
                    doubleQ = targetQvalues[
                        range(parameters.BATCH_SIZE), mainQaction]
                    targetQvalues = train_batch[2] + \
                        parameters.DISCOUNT * doubleQ * done_multiplier

                    feed_dict = {self.mainQNetwork.inputs: train_batch[0],
                                 self.mainQNetwork.Qtarget: targetQvalues,
                                 self.mainQNetwork.actions: train_batch[1]}
                    _ = self.sess.run(self.mainQNetwork.train,
                                      feed_dict=feed_dict)

                    update_target(self.update_target_ops, self.sess)

            if gif:
                self.env.save_gif("results/gif/save_{}.gif".format(self.n_gif))
                self.n_gif = (self.n_gif + 1) % 10

            if pre_training:
                self.best_run = max(self.best_run, episode_reward)

            else:
                DISPLAYER.add_reward(episode_reward)
                if episode_reward > self.best_run and \
                        self.total_steps > 500 + parameters.PRE_TRAIN_STEPS:
                    self.best_run = episode_reward
                    print("Save best", episode_reward)
                    SAVER.save('best', self.buffer)
                    self.play_gif('results/gif/best.gif')

            self.total_steps += 1

            if self.total_steps % 100 == 0:
                print("Episode", self.total_steps, ", reward", episode_reward)

            # Save the model
            if not pre_training and self.total_steps % 1000 == 0:
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
                                      feed_dict={self.mainQNetwork.inputs: [s]})
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
                                  feed_dict={self.mainQNetwork.inputs: [s]})
                a = a[0]
                s, r, done, info = self.env.act(a, gif=True)

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
