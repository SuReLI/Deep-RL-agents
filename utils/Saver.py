
import os
import tensorflow as tf

from settings import Settings


class Saver:
    """
    This class provides an easy way to save the weights of a Network and to load
    them from a saved file on disk.
    """

    def __init__(self, sess):
        self.sess = sess

    def save(self, n_episode):
        """
        Save all the tensorflow weights from a session into a file in a
        directory Settings.MODEL_PATH.

        Args:
            n_episode: the number of episodes the agent completed
                        This number is used for the save-file name.
        """
        print("Saving model", n_episode, "...")

        os.makedirs(os.path.dirname(Settings.MODEL_PATH), exist_ok=True)
        self.saver.save(self.sess, Settings.MODEL_PATH + "Model_" + str(n_episode) + ".ckpt")

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
                ckpt = tf.train.get_checkpoint_state(Settings.MODEL_PATH)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model loaded !\n")
                return True
            except (ValueError, AttributeError):
                print("No model is saved !\n")
                return False
        else:
            return False
