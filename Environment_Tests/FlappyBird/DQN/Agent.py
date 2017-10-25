
import random
import numpy as np
import tensorflow as tf
from collections import deque

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

        print("Creation of the main QNetwork...")
        self.mainQNetwork = QNetwork(self.state_size, self.action_size, 'main')
        print("Main QNetwork created !\n")

        print("Creation of the target QNetwork...")
        self.targetQNetwork = QNetwork(self.state_size, self.action_size,
                                       'target')
        print("Target QNetwork created !\n")

        self.buffer = PrioritizedReplayBuffer(parameters.BUFFER_SIZE,
                                              parameters.ALPHA)

        self.epsilon = parameters.EPSILON_START
        self.beta = parameters.BETA_START

        trainables = tf.trainable_variables()
        self.update_target_ops = updateTargetGraph(trainables)

        self.nb_ep = 1
        self.best_run = -1e10
        self.n_gif = 0

    def pre_train(self):
        print("Beginning of the pre-training...")

        for i in range(parameters.PRE_TRAIN_STEPS):

            s = self.env.reset()
            done = False
            episode_step = 0
            episode_reward = 0

            while episode_step < parameters.MAX_EPISODE_STEPS and not done:

                a = random.randint(0, self.action_size - 1)
                s_, r, done, info = self.env.act(a)
                self.buffer.add(s, a, r, s_, done)

                s = s_
                episode_reward += r
                episode_step += 1

            if i % 100 == 0:
                print("Pre-train step n", i)

            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def run(self):
        print("Beginning of the run...")

        self.pre_train()

        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < parameters.TRAINING_STEPS:

            s = self.env.reset()
            episode_reward = 0
            done = False

            memory = deque()
            discount_R = 0

            episode_step = 0
            max_step = parameters.MAX_EPISODE_STEPS + \
                self.nb_ep // parameters.EP_ELONGATION

            # Render parameters
            self.env.set_render(self.nb_ep % parameters.RENDER_FREQ == 0)
            gif = (self.nb_ep % parameters.GIF_FREQ ==
                   0) and parameters.DISPLAY

            while episode_step < max_step and not done:

                if random.random() < self.epsilon:
                    a = random.randint(0, self.action_size - 1)
                else:
                    a = self.sess.run(self.mainQNetwork.predict,
                                      feed_dict={self.mainQNetwork.inputs: [s]})
                    a = a[0]

                s_, r, done, info = self.env.act(a, gif)
                episode_reward += r

                memory.append((s, a, r, s_, done))

                if len(memory) <= parameters.N_STEP_RETURN:
                    discount_R += parameters.DISCOUNT**(len(memory) - 1) * r

                else:
                    s_mem, a_mem, r_mem, ss_mem, done_mem = memory.popleft()
                    discount_R = (discount_R - r_mem) / parameters.DISCOUNT +\
                        parameters.DISCOUNT_N * r
                    self.buffer.add(s_mem, a_mem, discount_R, s_, done)

                if episode_step % parameters.TRAINING_FREQ == 0:

                    train_batch = self.buffer.sample(parameters.BATCH_SIZE,
                                                     self.beta)
                    # Incr beta
                    if self.beta <= parameters.BETA_STOP:
                        self.beta += parameters.BETA_INCR

                    feed_dict = {self.mainQNetwork.inputs: train_batch[0]}
                    oldQvalues = self.sess.run(self.mainQNetwork.Qvalues,
                                               feed_dict=feed_dict)
                    tmp = [0] * len(oldQvalues)
                    for i, oldQvalue in enumerate(oldQvalues):
                        tmp[i] = oldQvalue[train_batch[1][i]]
                    oldQvalues = tmp

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
                    doubleQ = targetQvalues[range(parameters.BATCH_SIZE),
                                            mainQaction]
                    targetQvalues = train_batch[2] + \
                        parameters.DISCOUNT * doubleQ * done_multiplier

                    errors = np.abs(targetQvalues - oldQvalues)+1e-6
                    self.buffer.update_priorities(train_batch[6], errors)
                    
                    feed_dict = {self.mainQNetwork.inputs: train_batch[0],
                                 self.mainQNetwork.Qtarget: targetQvalues,
                                 self.mainQNetwork.actions: train_batch[1]}
                    _ = self.sess.run(self.mainQNetwork.train,
                                      feed_dict=feed_dict)

                    update_target(self.update_target_ops, self.sess)

                s = s_
                episode_step += 1
                self.total_steps += 1

            self.nb_ep += 1

            # Decay epsilon
            if self.epsilon > parameters.EPSILON_STOP:
                self.epsilon -= parameters.EPSILON_DECAY

            DISPLAYER.add_reward(episode_reward)
            if episode_reward > self.best_run and \
                    self.nb_ep > 50 + parameters.PRE_TRAIN_STEPS:
                self.best_run = episode_reward
                print("Save best", episode_reward)
                SAVER.save('best')
                self.play(1, 'results/gif/best.gif')

            if gif:
                self.env.save_gif('results/gif/gif_save', self.n_gif)
                self.n_gif = (self.n_gif + 1) % parameters.MAX_NB_GIF

            self.total_steps += 1

            if self.nb_ep % parameters.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %.3f'
                      ', Max steps: %i' % (self.nb_ep, episode_reward,
                                           episode_step, self.epsilon,
                                           max_step))

            # Save the model
            if self.nb_ep % parameters.SAVE_FREQ == 0:
                SAVER.save(self.nb_ep)

    def play(self, number_run, path=''):
        print("Playing for", number_run, "runs")

        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    a = self.sess.run(self.mainQNetwork.predict,
                                      feed_dict={self.mainQNetwork.inputs: [s]})
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
