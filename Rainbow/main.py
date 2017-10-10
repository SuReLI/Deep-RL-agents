
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

        SAVER.load(agent)

        print("Beginning of the run")
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        SAVER.save(agent.total_steps, agent.buffer)
        DISPLAYER.disp()

        agent.play(10)

        if parameters.GIF:
            agent.play_gif("results/gif/{}_1.gif".format(parameters.ENV))
            agent.play_gif("results/gif/{}_2.gif".format(parameters.ENV))
            agent.play_gif("results/gif/{}_3.gif".format(parameters.ENV))

    agent.stop()
