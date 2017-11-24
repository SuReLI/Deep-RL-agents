# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from Network import Network
from Environment import Environment

from Saver import SAVER
import settings

NB_PLAY = 1

if __name__ == '__main__':

    tf.reset_default_graph()

    device = "/cpu:0"
    stop_requested = False

    print("Creating the global network...")
    global_network = Network(0, device)
    env = Environment(True)

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    sess.run(tf.global_variables_initializer())

    SAVER.set_sess(sess)
    settings.LOAD = True
    global_total_time, wall_time, total_eps, total_steps = SAVER.load()

    for i in range(NB_PLAY):
        done = False
        reward, step = 0, 0

        state = env.reset()
        global_network.reset_state()

        while not done:

            pi, value = global_network.run_policy_and_value(sess, state)
            a = np.random.choice(settings.ACTION_SIZE, p=pi)
            state, r, done, _ = env.process(a)

            reward += r
            step += 1

        print("Episode reward {} (in {} steps)".format(reward, step))
