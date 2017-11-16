
import tensorflow as tf
import pickle
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
        try:
            os.makedirs(os.path.dirname("model/"))
        except OSError:
            pass
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".cptk")
        print("Model saved !")

    def load(self, agent, best=False):
        if parameters.LOAD:
            print("Loading model...")
            try:
                if best:
                    self.saver.restore(self.sess, 'model/Model_best.cptk')
                else:
                    ckpt = tf.train.get_checkpoint_state("model/")
                    print(ckpt.model_checkpoint_path)
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            except (ValueError, AttributeError):
                print("No model is saved !")
                self.sess.run(tf.global_variables_initializer())
            except tf.errors.InvalidArgumentError:
                print("The model save has not the same architecture !")
                self.sess.run(tf.global_variables_initializer())


        else:
            print("Initializing variables...")
            self.sess.run(tf.global_variables_initializer())
            print("Variables initialized !")


SAVER = Saver()
