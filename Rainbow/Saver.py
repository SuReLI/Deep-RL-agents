
import tensorflow as tf
import pickle

import parameters


class Saver:

    def __init__(self):
        pass

    def set_sess(self, sess):
        self.saver = tf.train.Saver()
        self.sess = sess

    def save(self, n_episode, agent_buffer):
        print("Saving model n", n_episode)
        os.makedirs(os.path.dirname("model/"), exist_ok=True)
        self.saver.save(self.sess, "model/Model_" + str(n_episode) + ".cptk")
        with open("model/buffer", "wb") as file:
            pickle.dump(agent_buffer, file)
        print("Model saved !")

    def load(self, agent):
        if parameters.LOAD:
            print("Loading model...")
            try:
                ckpt = tf.train.get_checkpoint_state("model/")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Loading buffer...")
                with open("model/buffer", "rb") as file:
                    agent.buffer = pickle.load(file)
                parameters.PRE_TRAIN_STEPS = 0
                parameters.EPSILON_START = parameters.EPSILON_STOP
                parameters.PRIOR_BETA_START = parameters.PRIOR_BETA_STOP
                print("Model loaded !")
            except (ValueError, AttributeError):
                print("No model is saved !")
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())


SAVER = Saver()
