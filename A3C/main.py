
# *********************************  TO DO  ********************************* #
#                                                                             #
# LSTM = False doesn't work (because of Agent which has LSTM everywhere)      #
#                                                                             #
# *************************************************************************** #


import tensorflow as tf
import parameters
import threading
from time import sleep

from Agent import Agent
from Saver import disp

if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.Session() as sess:

        with tf.device("/cpu:0"):

            render = parameters.DISPLAY
            master_agent = Agent(0, sess, render=render, master=True)

            workers = []
            for i in range(parameters.THREADS):
                workers.append(Agent(i + 1, sess, render=False))

        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for i, worker in enumerate(workers):
            print("Threading worker", i + 1)
            work = lambda: worker.work(sess, coord)
            t = threading.Thread(target=(work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        try:
            coord.join(worker_threads)
        except Exception as e:
            coord.request_stop(e)
        except KeyboardInterrupt as e:
            coord.request_stop()
        finally:
            sleep(1)
            print("End of the training")

        disp()
        master_agent.play(sess, 10)
