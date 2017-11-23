
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER
import parameters

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        agent = Agent(sess)

        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")

        DISPLAYER.disp()

        agent.play(5)

    agent.stop()
