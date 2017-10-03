
import tensorflow as tf
import parameters
import threading
from time import sleep

from Agent import Agent
from MasterNetwork import Network

if __name__ == '__main__':

    tf.reset_default_graph()

    with tf.device("/cpu:0"):

        master_agent = Agent(0, master=True, render=True)
        master_network = Network(master_agent.env.get_state_dims(),
                                 master_agent.env.get_action_size(),
                                 'global')

        sleep(1)
        workers = []
        for i in range(parameters.THREADS):
            workers.append(Agent(i+1, master=False, render=True))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for i, worker in enumerate(workers):
            print("Start worker", i)
            work = lambda: worker.run(sess)
            t = threading.Thread(target=(work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
