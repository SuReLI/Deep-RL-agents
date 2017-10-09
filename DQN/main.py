
import os
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER
from Saver import SAVER
import parameters


if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        agent = Agent(sess)
        SAVER.set_sess(sess)

        SAVER.load()

        print("Beginning of the run")
        agent.run()
        DISPLAYER.disp()

        agent.play()
        agent.stop()
