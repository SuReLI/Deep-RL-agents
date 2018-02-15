
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

        gui = GUI.Interface(['ep_reward', 'plot', 'plot_distrib', 'render', 'gif', 'save'])
        gui_thread = threading.Thread(target=gui.run)

        agent = Agent(sess, gui, displayer, saver)

        saver.load(agent)

        gui_thread.start()
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        
        saver.save(agent.nb_ep)
        displayer.disp()

        gui_thread.join()
        # agent.play(5)

    agent.stop()
