
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER

import parameters

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)

        print("Beginning of the run")
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        DISPLAYER.disp()

        agent.play(5)

    agent.close()
