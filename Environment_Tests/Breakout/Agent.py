# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from collections import deque

from Environment import Environment
from Network import Network

from Displayer import DISPLAYER
from settings import *


class Agent:

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input

        self.local_network = Network(thread_index, device)
        self.local_network.build_loss()

        with tf.device(device):
            local_var_refs = [v._ref() for v in self.local_network.get_vars()]

            self.gradients = tf.gradients(self.local_network.total_loss,
                                          local_var_refs,
                                          gate_gradients=False,
                                          aggregation_method=None,
                                          colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.update_network = self.local_network.copy_network(global_network)

        self.env = Environment(thread_index == 1 and DISPLAY)
        self.state = self.env.reset()

        self.worker_total_steps = 0
        self.worker_total_eps = 0
        self.start_time = time.time()

        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (MAX_TIME_STEP - global_time_step) / MAX_TIME_STEP
        return max(learning_rate, 0)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, total_steps):
        summary_str = sess.run(summary_op, feed_dict={score_input: score})
        summary_writer.add_summary(summary_str, total_steps)
        summary_writer.flush()

    def process(self, sess, total_steps, summary_writer, summary_op, score_input):

        start_time = time.time()
        buffer = []
        done = False
        episode_step = 0

        # copy weights from global to local
        sess.run(self.update_network)

        start_lstm_state = self.local_network.lstm_state_out

        for i in range(UPDATE_FREQ):

            pi, value = self.local_network.run_policy_and_value(sess,
                                                                self.state)

            a = np.random.choice(ACTION_SIZE, p=pi)
            s_, r, terminal, _ = self.env.process(a)

            self.episode_reward += r

            # clip reward
            r = np.clip(r, -1, 1)
            buffer.append((self.state, a, r, value))

            episode_step += 1
            self.worker_total_steps += 1
            self.state = s_

            if terminal:
                done = True
                self.worker_total_eps += 1

                DISPLAYER.add_reward(self.episode_reward, self.thread_index)

                if (self.thread_index == 1 and
                        self.worker_total_eps % DISP_REWARD_FREQ == 0):
                    cur_learning_rate = self._anneal_learning_rate(total_steps)
                    print('Episode %i, Reward %i, Steps %i, LR %g' %
                          (self.worker_total_eps, self.episode_reward,
                           episode_step, cur_learning_rate))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, total_steps)

                self.episode_reward = 0
                self.env.reset()
                self.local_network.reset_state()

                render = (DISPLAY and self.thread_index == 1 and
                          (self.worker_total_eps - 1) % RENDER_FREQ == 0)
                self.env.set_render(render)

                break

        batch_s = deque()
        batch_a = deque()
        batch_td = deque()
        batch_R = deque()

        # Bootstrapping
        R = 0.0
        if not done:
            R = self.local_network.run_value(sess, self.state)

        # compute and accumulate gradients
        for i in range(len(buffer) - 1, -1, -1):
            si, ai, ri, Vi = buffer[i]
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_s.appendleft(si)
            batch_a.appendleft(a)
            batch_td.appendleft(td)
            batch_R.appendleft(R)

        cur_learning_rate = self._anneal_learning_rate(total_steps)

        feed_dict = {self.local_network.state: batch_s,
                     self.local_network.action: batch_a,
                     self.local_network.td_error: batch_td,
                     self.local_network.reward: batch_R,
                     self.local_network.initial_lstm_state: start_lstm_state,
                     self.local_network.step_size: [len(batch_a)],
                     self.learning_rate_input: cur_learning_rate}
        sess.run(self.apply_gradients, feed_dict=feed_dict)

        if done and (self.thread_index == 1) and \
                (self.worker_total_eps % PERF_FREQ == 0 or
                 self.worker_total_eps == 15):
            global_time = time.time() - self.start_time
            steps_per_sec = total_steps / global_time
            print("### Performance : {} STEPS in {:.0f} sec."
                  "{:.0f} STEPS/sec. {:.2f}M STEPS/hour ###".format(
                      total_steps,  global_time, steps_per_sec,
                      steps_per_sec * 3600 / 1000000.))

        elapsed_time = time.time() - start_time
        return elapsed_time, done, episode_step

    def close(self):
        self.env.close()
