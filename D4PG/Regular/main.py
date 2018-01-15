
import tensorflow as tf
import threading
import time

import Actor
# from Actor import Actor, request_stop, STOP_REQUESTED
from Learner import Learner
from Displayer import DISPLAYER
import settings


def run_gui():
    print("Running GUI")
    import GUI
    GUI.main()


def main():

    tf.reset_default_graph()

    with tf.Session() as sess:

        workers = []
        for i in range(settings.NB_ACTORS):
            workers.append(Actor.Actor(sess, i + 1))

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

        GUI_thread = threading.Thread(target=run_gui)

        sess.run(tf.global_variables_initializer())

        for t in threads:
            t.start()
        GUI_thread.start()
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

        GUI_thread.join()


if __name__ == '__main__':
    main()
