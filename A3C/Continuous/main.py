
###############################################################################
# To try :
#   - Update global vars <= tau * glob_var + (1-tau) * loc_var
#   - Bootstrap after
#   - Clip the gradient if relu
#
#
###############################################################################




import tensorflow as tf
import parameters
import threading
from time import sleep

from Agent import Agent
from Displayer import DISPLAYER


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

        sess.run(tf.global_variables_initializer())

        # Run threads that each contains one worker
        worker_threads = []
        for i, worker in enumerate(workers):
            print("Threading worker", i + 1)
            sleep(0.1)
            work = lambda: worker.work(sess, coord)
            t = threading.Thread(target=(work))
            t.start()
            worker_threads.append(t)
            sleep(0.1)

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
        master_agent.test(sess, 10)
