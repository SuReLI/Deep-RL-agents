
import tensorflow as tf
import parameters
import threading
from time import sleep

from Agent import Agent
from MasterNetwork import Network

if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.device("/cpu:0"):

        master_agent = Agent(0, render=True, master=True)

        workers = []
        for i in range(1):  #parameters.THREADS):
            workers.append(Agent(i+1, render=False))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for i, worker in enumerate(workers):
            print("Start worker", i)
            work = lambda: worker.work(sess, coord)
            t = threading.Thread(target=(work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
