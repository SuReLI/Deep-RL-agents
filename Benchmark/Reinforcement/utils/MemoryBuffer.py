
import tensorflow as tf
import numpy as np
import random
from collections import deque

from settings import Settings


class MemoryBuffer:

    def __init__(self, sess, coord):
        print("Initializing MemoryBuffer...")

        self.sess = sess
        self.coord = coord

        self.buffer_s = np.empty([Settings.BUFFER_SIZE, *Settings.STATE_SIZE], dtype=np.float32)
        self.buffer_a = np.empty([Settings.BUFFER_SIZE, Settings.ACTION_SIZE], dtype=np.float32)
        self.buffer_r = np.empty([Settings.BUFFER_SIZE], dtype=np.float32)
        self.buffer_s_ = np.empty([Settings.BUFFER_SIZE, *Settings.STATE_SIZE], dtype=np.float32)
        self.buffer_d = np.empty([Settings.BUFFER_SIZE], dtype=np.float32)

        self.index = 0

        shapes = (Settings.STATE_SIZE, Settings.ACTION_SIZE, (), Settings.STATE_SIZE, ())

        with self.sess.as_default(), self.sess.graph.as_default():
            queue = tf.FIFOQueue(capacity=3*Settings.BATCH_SIZE,
                                 dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                 shapes=shapes)

            idx = tf.random_uniform(dtype=tf.int32, minval=0, maxval=1000, shape=[Settings.BATCH_SIZE])

            print(tf.gather(self.buffer_s, idx))
            print(tf.gather(self.buffer_d, idx))
            print(tf.concat([tf.gather(self.buffer_s, idx), tf.gather(self.buffer_d, idx)], axis=1))
            get_data = tf.concat([tf.gather(b, idx) for b in (self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_s_, self.buffer_d)], axis=1)


            # get_data = tf.py_func(self.get_batch, inp=[], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])


            enqueue_op = queue.enqueue_many(get_data)

            qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
            tf.train.add_queue_runner(qr)

            self.dequeue = queue.dequeue_many(Settings.BATCH_SIZE)
            self.size = queue.size()

        print("MemoryBuffer initialized !\n")

    # def get_batch(self):
    #     idx = random.sample(range(len(self.buffer)), min(Settings.BATCH_SIZE, len(self.buffer)))
    #     # batch = random.sample(self.buffer, min(Settings.BATCH_SIZE, len(self.buffer)))

    #     for i in idx:
    #         self.idx[i] += 1

    #     qs = np.asarray([self.buffer[i][0] for i in idx], dtype=np.float32)
    #     qa = np.asarray([self.buffer[i][1] for i in idx], dtype=np.float32)
    #     qr = np.asarray([self.buffer[i][2] for i in idx], dtype=np.float32)
    #     qs_ = np.asarray([self.buffer[i][3] for i in idx], dtype=np.float32)
    #     qdone = np.asarray([self.buffer[i][4] for i in idx], dtype=np.float32)

    #     return [qs, qa, qr, qs_, qdone]

    def add(self, s, a, r, s_, d):
    	self.buffer_s[self.index] = s
    	self.buffer_a[self.index] = a
    	self.buffer_r[self.index] = r
    	self.buffer_s_[self.index] = s_
    	self.buffer_d[self.index] = d
    	self.index = (self.index + 1) % Settings.BUFFER_SIZE

        # n = len(self.buffer)
        # if n % 1000 == 0:
            # print("Buffer size : ", n)
            # print("% seen : ", sum(self.idx[:n]) / n)
            # print()
