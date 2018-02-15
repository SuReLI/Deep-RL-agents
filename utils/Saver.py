
import tensorflow as tf
import os

from settings import Settings


class Saver:

    def __init__(self, sess):
        self.sess = sess

    def save(self, n_episode):
        print("Saving model", n_episode, "...")
        os.makedirs(os.path.dirname("model/"), exist_ok=True)
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".cptk")
        print("Model saved !")

    def load(self, agent):
        self.saver = tf.train.Saver()
        if Settings.LOAD:
            print("Loading model...")
            try:
                ckpt = tf.train.get_checkpoint_state("model/")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            except (ValueError, AttributeError):
                print("No model is saved !")
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())
