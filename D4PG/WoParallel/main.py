
import tensorflow as tf
import threading

from Agent import Agent

from Displayer import DISPLAYER
import GUI

import settings

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)
        GUI_thread = threading.Thread(target=GUI.main)

        sess.run(tf.global_variables_initializer())

        GUI_thread.start()
        print("Beginning of the run")
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")
        DISPLAYER.disp()

        GUI_thread.join()

    agent.close()
