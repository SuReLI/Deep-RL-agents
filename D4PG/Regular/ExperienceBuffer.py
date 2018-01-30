
import tensorflow as tf

from settings import *


class MemoryBuffer:

    def __init__(self, sess):

        self.sess = sess

        with self.sess.as_default(), self.sess.graph.as_default():

            min_after_dequeue = 5 + (NB_ACTORS + 1) * BATCH_SIZE

            shapes = (STATE_SIZE, ACTION_SIZE, (), STATE_SIZE, ())
            shapes = list(map(tf.TensorShape, shapes))

            self.queue = tf.RandomShuffleQueue(capacity=MEMORY_SIZE,
                                      min_after_dequeue=min_after_dequeue,
                                      dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                      shapes=shapes)

            self.qs = tf.placeholder(tf.float32, [*STATE_SIZE])
            self.qa = tf.placeholder(tf.float32, [ACTION_SIZE])
            self.qr = tf.placeholder(tf.float32, ())
            self.qs_ = tf.placeholder(tf.float32, [*STATE_SIZE])
            self.qdone = tf.placeholder(tf.float32, ())

            self.enqueue_op = self.queue.enqueue([self.qs, self.qa, self.qr, self.qs_, self.qdone])

            self.dequeue = self.queue.dequeue_many(BATCH_SIZE)

    def enqueue(self, s, a, r, s_, d):

        feed_dict = {self.qs: s, self.qa: a, self.qr: r, self.qs_: s_, self.qdone: d}
        with self.sess.as_default(), self.sess.graph.as_default():
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)

    def __len__(self):
        return self.sess.run(self.queue.size)
