
import threading
import tensorflow as tf

from Agent import Agent

import GUI
import Displayer
import Saver

from settings import Settings

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        settings = Settings()
        saver = Saver.Saver(settings, sess)
        displayer = Displayer.Displayer(settings)

        gui = GUI.Interface(settings, ['ep_reward', 'plot', 'render', 'gif', 'save'])
        gui_thread = threading.Thread(target=gui.run)

        agent = Agent(settings, sess, gui, displayer, saver)

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
        # agent.play(3)

    agent.stop()
