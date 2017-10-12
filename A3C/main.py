
import tensorflow as tf
import parameters
import threading
from time import sleep

from Agent import Agent
from Displayer import DISPLAYER
from Saver import SAVER


if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.Session() as sess:

        with tf.device("/cpu:0"):

            # Create the global network
            render = parameters.DISPLAY
            master_agent = Agent(0, sess, render=render, master=True)

            # Create all the workers
            workers = []
            for i in range(parameters.THREADS):
                workers.append(Agent(i + 1, sess, render=False))

        coord = tf.train.Coordinator()
        SAVER.set_sess(sess)

        SAVER.load()

        # Run threads that each contains one worker
        worker_threads = []
        for i, worker in enumerate(workers):
            print("Threading worker", i + 1)
            work = lambda: worker.work(sess, coord)
            t = threading.Thread(target=(work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        try:
            # Wait till all the workers are done
            coord.join(worker_threads)
        except Exception as e:
            coord.request_stop(e)
        except KeyboardInterrupt as e:
            coord.request_stop()
        finally:
            sleep(1)
            print("End of the training")

        DISPLAYER.disp_all()
        DISPLAYER.disp_one()
        DISPLAYER.disp_seq()
        master_agent.play(sess, 10)
        master_agent.play_gif(sess,
                              "results/gif/{}_1.gif".format(parameters.ENV))
