
import sys
sys.path.append("./utils")

import threading
import tensorflow as tf
import time

from Agent import Agent
from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer

import GUI

from settings import Settings


if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.Session() as sess:

        buffer = ExperienceBuffer()

        gui = GUI.Interface(['ep_reward'])
        gui_thread = threading.Thread(target=gui.run)

        threads = []
        for i in range(Settings.NB_ACTORS):
            agent = Agent(sess, i, gui, buffer)
            threads.append(threading.Thread(target=agent.run))

        with tf.device('/device:GPU:0'):
            learner = QNetwork(sess, gui, buffer)
        threads.append(threading.Thread(target=learner.run))

        sess.run(tf.global_variables_initializer())

        gui_thread.start()
        for t in threads:
            t.start()

        print("Running...")

        try:
            while not gui.STOP:
                time.sleep(1)

        except:
            pass

        gui.STOP = True

        for t in threads:
            t.join()

        gui_thread.join()
