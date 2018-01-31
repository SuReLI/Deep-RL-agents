
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

        shapes = (STATE_SIZE, ACTION_SIZE, (), STATE_SIZE, ())

        with self.sess.as_default(), self.sess.graph.as_default():
            queue = tf.FIFOQueue(capacity=10*BATCH_SIZE,
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
        batch = random.sample(self.buffer, min(BATCH_SIZE, len(self.buffer)))

        qs = np.asarray([elem[0] for elem in batch], dtype=np.float32)
        qa = np.asarray([elem[1] for elem in batch], dtype=np.float32)
        qr = np.asarray([elem[2] for elem in batch], dtype=np.float32)
        qs_ = np.asarray([elem[3] for elem in batch], dtype=np.float32)
        qdone = np.asarray([elem[4] for elem in batch], dtype=np.float32)

        print("Add : ", self.sess.run(self.size))

        return [qs, qa, qr, qs_, qdone]

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))
