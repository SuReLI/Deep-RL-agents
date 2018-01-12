
import tensorflow as tf
import os

import settings


class Saver:

    def set_sess(self, sess):
        self.saver = tf.train.Saver()
        self.sess = sess

    def save(self, n_episode):
        print("Saving model", n_episode, "...")
        os.makedirs(os.path.dirname("model/"), exist_ok=True)
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".cptk")
        print("Model saved !")

    def load(self, agent):
        if settings.LOAD:
            print("Loading model...")
            try:
                ckpt = tf.train.get_checkpoint_state("model/")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            except (ValueError, AttributeError):
                print("No model is saved !")
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())


SAVER = Saver()
