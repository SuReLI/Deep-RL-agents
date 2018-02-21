
import random
import numpy as np
import tensorflow as tf
from collections import deque

from Environment import Environment
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from QNetwork import QNetwork
from Model import copy_vars
from settings import Settings


class Agent:

    def __init__(self, sess, gui, displayer, saver):
        print("Initializing the agent...")

        self.sess = sess
        self.gui = gui
        self.displayer = displayer
        self.saver = saver

        self.env = Environment()

        self.mainQNetwork = QNetwork(sess, 'main')
        self.targetQNetwork = QNetwork(sess, 'target')

        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NB_ATOMS - 1)
        self.z = np.linspace(Settings.MIN_Q, Settings.MAX_Q, Settings.NB_ATOMS)

        self.buffer = PrioritizedReplayBuffer(Settings.BUFFER_SIZE,
                                              Settings.ALPHA)

        self.epsilon = Settings.EPSILON_START
        self.beta = Settings.BETA_START

        self.init_update_target = copy_vars(self.mainQNetwork.vars,
                                            self.targetQNetwork.vars,
                                            1, 'init_update_target')

        self.update_target = copy_vars(self.mainQNetwork.vars,
                                       self.targetQNetwork.vars,
                                       Settings.UPDATE_TARGET_RATE,
                                       'update_target')

        self.nb_ep = 1
        self.best_run = -1e10
        self.n_gif = 0

    def pre_train(self):
        print("Beginning of the pre-training...")

        for i in range(Settings.PRE_TRAIN_STEPS):

            s = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while episode_step < Settings.MAX_EPISODE_STEPS and not done:

                a = self.env.act_random()
                s_, r, done, info = self.env.act(a)
                self.buffer.add(s, a, r, s_, 0 if done else 1)

                s = s_
                episode_reward += r
                episode_step += 1

            if Settings.PRE_TRAIN_STEPS > 5 and i % (Settings.PRE_TRAIN_STEPS // 5) == 0:
                print("Pre-train step n", i)

            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def learn(self):

        states, actions, rewards, next_states, not_done, weights, idx = self.buffer.sample(Settings.BATCH_SIZE, self.beta)

        Qdistrib = self.mainQNetwork(states)            # P(s_t, .)
        Qdistrib_main_action = Qdistrib[range(Settings.BATCH_SIZE), actions] # P(s_t, a_t)

        Qdistrib_next = self.mainQNetwork(next_states)  # P(s_{t+n}, .)
        Qvalue_next = np.mean(self.z * Qdistrib_next, axis=2)
        best_action = np.argmax(Qvalue_next, axis=1)  # argmax_a Q(s_{t+n}, a)
        Qdistrib_next_target = self.targetQNetwork(next_states)  # P_target(s_{t+n}, .)
        Qdistrib_next_target_best_action = Qdistrib_next_target[range(Settings.BATCH_SIZE), best_action]   # P_target(s_{t+n}, best_action)

        Tz = rewards[:, np.newaxis] + Settings.DISCOUNT_N * np.outer(not_done, self.z[np.newaxis])
        Tz = np.clip(Tz, Settings.MIN_Q, Settings.MAX_Q - 1e-5)

        b = (Tz - Settings.MIN_Q) / self.delta_z
        l = np.floor(b).astype(int)
        u = l + 1

        m = np.zeros([Settings.BATCH_SIZE, Settings.NB_ATOMS])
        for j in range(Settings.NB_ATOMS):
            m[range(Settings.BATCH_SIZE), l[:, j]] += Qdistrib_next_target_best_action[:, j] * (u[:, j] - b[:, j])
            m[range(Settings.BATCH_SIZE), u[:, j]] += Qdistrib_next_target_best_action[:, j] * (b[:, j] - l[:, j])

        loss = -np.sum(m * np.log(Qdistrib_main_action + 1e-10), axis=1)

    def run(self):
        print("Beginning of the run...")

        self.pre_train()
        self.sess.run(self.init_update_target)

        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < Settings.TRAINING_STEPS and not self.gui.STOP:

            s = self.env.reset()
            episode_reward = 0
            done = False
            memory = deque()

            episode_step = 1
            max_step = Settings.MAX_EPISODE_STEPS
            if Settings.EP_ELONGATION > 0:
                max_step += self.nb_ep // Settings.EP_ELONGATION

            # Render settings
            self.env.set_render(self.gui.render.get(self.nb_ep))
            self.env.set_gif(self.gui.gif.get(self.nb_ep))

            while episode_step <= max_step and not done:

                if True:#random.random() < self.epsilon:
                    a = random.randint(0, Settings.ACTION_SIZE - 1)
                else:
                    a, = self.sess.run(self.mainQNetwork.predict,
                                       feed_dict={self.mainQNetwork.inputs: [s]})

                s_, r, done, info = self.env.act(a)
                episode_reward += r

                memory.append((s, a, r, s_, 0 if done else 1))

                if len(memory) > Settings.N_STEP_RETURN:
                    s_mem, a_mem, discount_R, ss_mem, done_mem = memory.popleft()
                    for i, (si, ai, ri, s_i, di) in enumerate(memory):
                        discount_R += ri * Settings.DISCOUNT ** (i + 1)
                    self.buffer.add(s_mem, a_mem, discount_R, s_, 0 if done else 1)

                if episode_step % Settings.TRAINING_FREQ == 0:
                    self.learn()
                    self.sess.run(self.update_target)

                s = s_
                episode_step += 1
                self.total_steps += 1

            self.nb_ep += 1

            # Decay epsilon
            if self.epsilon > Settings.EPSILON_STOP:
                self.epsilon -= Settings.EPSILON_DECAY

            self.displayer.add_reward(episode_reward, self.gui.plot.get(self.nb_ep))
            # if episode_reward > self.best_run and \
            #         self.nb_ep > 50 + Settings.PRE_TRAIN_STEPS:
            #     self.best_run = episode_reward
            #     print("Save best", episode_reward)
            #     SAVER.save('best')
            #     self.play(1, 'results/gif/best.gif')

            if self.gui.ep_reward.get(self.nb_ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %i'
                      ', Max steps: %i' % (self.nb_ep, episode_reward,
                                           episode_step, self.epsilon,
                                           max_step))

            # Save the model
            if self.gui.save.get(self.nb_ep):
                self.saver.save(self.nb_ep)

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
