
import os
import sys
s = "/".join(os.getcwd().split("/")[:-1]) + '/utils'
sys.path.append(s)                  # Include utils module

import tensorflow as tf

from Agent import Agent

import GUI
import Saver
import Displayer

from settings import Settings

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        saver = Saver.Saver(sess)
        displayer = Displayer.Displayer()

        gui = GUI.Interface(['ep_reward', 'plot', 'render', 'gif', 'save'])

        agent = Agent(sess, gui, displayer, saver)

        if not saver.load():
            sess.run(tf.global_variables_initializer())

        agent.run()

        saver.save(agent.total_steps)
        agent.display()

        # agent.play(5)

    agent.stop()
