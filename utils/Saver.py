
"""
This class provides an easy way to save the weights of a Network and to load
them from a saved file on disk.
"""

import tensorflow as tf
import os

from settings import Settings


class Saver:

    def __init__(self, sess):
        self.sess = sess

    def save(self, n_episode):
        """
        Save all the tensorflow weights from a session into a file in a
        directory "model/".

        Args:
            n_episode: the number of episodes the agent completed
                        This number is used for the save-file name.
        """
        print("Saving model", n_episode, "...")

        os.makedirs(os.path.dirname("model/"), exist_ok=True)
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".cptk")

        print("Model saved !")

    def load(self):
        """
        Try to restore the weights to the current session.
        If the LOAD setting is False or if no model is saved, this methods just
        run an initialization of tensorflow variables.
        """
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
