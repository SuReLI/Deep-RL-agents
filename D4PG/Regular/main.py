
import tensorflow as tf
import threading
import time

import Actor
from Learner import Learner
from ExperienceBuffer import MemoryBuffer
import GUI
from Displayer import DISPLAYER
from settings import *


if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.Session() as sess:

        queue = MemoryBuffer(sess)

        workers = []
        for i in range(NB_ACTORS):
            workers.append(Actor.Actor(sess, i + 1, queue))

        # with tf.device('/device:GPU:0'):
        print("Initializing learner...")
        learner = Learner(sess, queue)
        print("Learner initialized !\n")

        sess.run(tf.global_variables_initializer())
        for worker in workers:
            worker.build_update()

        if LOAD:
            learner.load()

        sess.graph.finalize()

        threads = []
        for i in range(NB_ACTORS):
            thread = threading.Thread(target=workers[i].run)
            threads.append(thread)

        threads.append(threading.Thread(target=learner.run))

        if INTERFACE:
            GUI_thread = threading.Thread(target=GUI.main)
            GUI_thread.start()


        for t in threads:
            t.start()
        print("Running...")

        try:
            while not Actor.STOP_REQUESTED:
                time.sleep(1)
        except KeyboardInterrupt:
            Actor.request_stop()

        for t in threads:
            t.join()

        learner.save()

        DISPLAYER.disp()
        DISPLAYER.disp_q()

        if INTERFACE:
            GUI_thread.join()
