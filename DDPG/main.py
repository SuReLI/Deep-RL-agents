
import os
import sys
s = os.getcwd()
sys.path.append(s[:s.find('RL-Agents') + 10] + 'utils')  # Include utils module

import threading
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
        gui_thread = threading.Thread(target=gui.run)

        agent = Agent(sess, gui, displayer, saver)

        if not saver.load():
            sess.run(tf.global_variables_initializer())

        gui_thread.start()
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")

        saver.save(agent.total_steps)
        displayer.disp()

        gui_thread.join()
        # agent.play(5)

    agent.stop()
