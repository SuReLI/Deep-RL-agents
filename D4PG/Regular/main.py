
import tensorflow as tf
import threading
import time

import Actor
from Learner import Learner
from MemoryBuffer import MemoryBuffer
import GUI
from Displayer import DISPLAYER
from settings import *


if __name__ == '__main__':

    tf.reset_default_graph()

    coord = tf.train.Coordinator()

    with tf.Session() as sess:

        buffer = MemoryBuffer(sess, coord)

        # Initialize learner and learner thread
        learner = Learner(sess, coord, buffer)
        threads = [threading.Thread(target=learner.run)]

        # Initialize workers and worker threads
        workers = []
        for i in range(NB_ACTORS):
            worker = Actor.Actor(sess, coord, i + 1, buffer)
            workers.append(worker)
            threads.append(threading.Thread(target=worker.run))

        if not learner.load():
            sess.run(tf.global_variables_initializer())

        sess.graph.finalize()

        if INTERFACE:
            GUI_thread = threading.Thread(target=GUI.main, args=(coord,))
            GUI_thread.start()

        for t in threads:
            t.start()

        print("Running...")

        time.sleep(3)
        queue_threads = tf.train.start_queue_runners(coord=coord)

        try:
            coord.join(threads + queue_threads)
        except Exception as e:
            coord.request_stop(e)
        except KeyboardInterrupt:
            coord.request_stop()
        print("End of the run !")

        learner.save()

        DISPLAYER.disp()
        DISPLAYER.disp_q()

        if INTERFACE:
            GUI_thread.join()
