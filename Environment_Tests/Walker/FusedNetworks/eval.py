
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER
from Saver import SAVER

import parameters

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)
        SAVER.set_sess(sess)

        SAVER.load(agent, best=True)

        agent.play(1, "results/gif/".format(parameters.ENV))

    agent.close()
