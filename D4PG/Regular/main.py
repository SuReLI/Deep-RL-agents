
import tensorflow as tf
import threading
import signal

from Actor import Actor, request_stop
from Learner import Learner
from Displayer import DISPLAYER
import settings


def main():

    tf.reset_default_graph()

    with tf.Session() as sess:

        workers = []
        for i in range(settings.NB_ACTORS):
            workers.append(Actor(sess, i + 1))

        # with tf.device('/device:GPU:0'):
        print("Initializing learner...")
        learner = Learner(sess, *workers[0].get_env_features())
        print("Learner initialized !\n")
        if settings.LOAD:
            learner.load()

        threads = []
        for i in range(settings.NB_ACTORS):
            thread = threading.Thread(target=workers[i].run)
            threads.append(thread)

        threads.append(threading.Thread(target=learner.run))

        sess.run(tf.global_variables_initializer())

        for t in threads:
            t.start()
        print("Running...")

        signal.signal(signal.SIGINT, request_stop)
        signal.pause()

        for t in threads:
            t.join()

        learner.save()

        DISPLAYER.disp()
        DISPLAYER.disp_q()


if __name__ == '__main__':
    main()
