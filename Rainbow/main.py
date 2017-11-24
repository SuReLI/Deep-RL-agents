
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

        SAVER.load(agent)

        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        SAVER.save(agent.nb_ep)
        DISPLAYER.disp()

        agent.play(10)

        agent.play(3, "results/gif/{}".format(parameters.ENV))

    agent.stop()
