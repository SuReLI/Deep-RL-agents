
import tensorflow as tf
import os

import parameters


class Saver:

    def __init__(self):
        pass

    def set_sess(self, sess):
        self.saver = tf.train.Saver()
        self.sess = sess

    def save(self, n_episode):
        print("Saving model", n_episode, "...")
        os.makedirs(os.path.dirname("model/"), exist_ok=True)
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".ckpt")
        print("Model saved !")

    def load(self, agent, best=False):
        if parameters.LOAD:
            print("Loading model...")
            try:
                ckpt = tf.train.get_checkpoint_state("model/")
                if best:
                    self.saver.restore(self.sess, 'model/Model_best.ckpt')
                else:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model loaded !")
            except (ValueError, AttributeError):
                print("No model is saved !")
                self.sess.run(tf.global_variables_initializer())
            print()
        else:
            self.sess.run(tf.global_variables_initializer())


SAVER = Saver()
