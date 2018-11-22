
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

        features = ['ep_reward', 'plot', 'render', 'gif', 'save']
        if Settings.DISTRIBUTIONAL:
            features.append('plot_distrib')
        gui = GUI.Interface(features)

        agent = Agent(sess, gui, displayer, saver)

        if not saver.load():
            sess.run(tf.global_variables_initializer())

        agent.run()

        saver.save(agent.nb_ep)
        agent.display()

        # agent.play(3)

    agent.stop()
