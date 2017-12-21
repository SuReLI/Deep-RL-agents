
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER
from Saver import SAVER

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)
        SAVER.set_sess(sess)

        SAVER.load(agent)

        print("Beginning of the run")
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        SAVER.save(agent.nb_ep)
        DISPLAYER.disp()

        # agent.play(10)

    agent.close()


def test(best=False):
    with tf.Session() as sess:
        agent = Agent(sess)
        agent.sess = sess
        SAVER.set_sess(sess)
        SAVER.load(agent, best)
        agent.env.set_render(True)
        agent.play(3)