
import threading
import tensorflow as tf

from Agent import Agent

import GUI
from Displayer import DISPLAYER
from Saver import SAVER

import settings

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)
        SAVER.set_sess(sess)

        SAVER.load(agent)

        if settings.GUI:
            gui = threading.Thread(target=GUI.main)
            gui.start()

        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        SAVER.save(agent.nb_ep)
        DISPLAYER.disp()

        if settings.GUI:
            gui.join()
        else:
            agent.play(5)

    agent.stop()
