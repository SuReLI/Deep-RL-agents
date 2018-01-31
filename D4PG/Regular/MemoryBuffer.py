
import tensorflow as tf
import numpy as np
import random
from collections import deque

from settings import *


class MemoryBuffer:

    def __init__(self, sess, coord):
        print("Initializing MemoryBuffer...")

        self.sess = sess
        self.coord = coord
        self.buffer = deque(maxlen=MEMORY_SIZE)
        self.idx = [0] * MEMORY_SIZE

        shapes = (STATE_SIZE, ACTION_SIZE, (), STATE_SIZE, ())

        with self.sess.as_default(), self.sess.graph.as_default():
            queue = tf.FIFOQueue(capacity=3*BATCH_SIZE,
                                 dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                 shapes=shapes)

            get_data = tf.py_func(self.get_batch, inp=[], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            enqueue_op = queue.enqueue_many(get_data)

            qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
            tf.train.add_queue_runner(qr)

            self.dequeue = queue.dequeue_many(BATCH_SIZE)
            self.size = queue.size()

        print("MemoryBuffer initialized !\n")

    def get_batch(self):
        idx = random.sample(range(len(self.buffer)), min(BATCH_SIZE, len(self.buffer)))
        # batch = random.sample(self.buffer, min(BATCH_SIZE, len(self.buffer)))

        for i in idx:
            self.idx[i] += 1

        qs = np.asarray([self.buffer[i][0] for i in idx], dtype=np.float32)
        qa = np.asarray([self.buffer[i][1] for i in idx], dtype=np.float32)
        qr = np.asarray([self.buffer[i][2] for i in idx], dtype=np.float32)
        qs_ = np.asarray([self.buffer[i][3] for i in idx], dtype=np.float32)
        qdone = np.asarray([self.buffer[i][4] for i in idx], dtype=np.float32)

        return [qs, qa, qr, qs_, qdone]

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

        # n = len(self.buffer)
        # if n % 1000 == 0:
            # print("Buffer size : ", n)
            # print("% seen : ", sum(self.idx[:n]) / n)
            # print()
