
import tensorflow as tf
import threading
import time

from tensorflow.python.client import timeline

import Actor
import GUI
from Learner import Learner
from Displayer import DISPLAYER
import settings


class Sess(tf.Session):
    def __init__(self, options, meta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = options
        self.meta = meta
    def run(self, *args, **kwargs):
        return super().run(options=self.op, run_metadata=self.meta, *args, **kwargs)


if __name__ == '__main__':


    tf.reset_default_graph()

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    meta = tf.RunMetadata()
    config = tf.ConfigProto(log_device_placement=True,
                            device_count={"CPU":10, "GPU":1},
                            inter_op_parallelism_threads=10)
    with Sess(options, meta, config=config) as sess:

        workers = []
        for i in range(settings.NB_ACTORS):
            with tf.device("/device:CPU:"+str(i)):
                workers.append(Actor(sess, i + 1))

        print("Initializing learner...")
        with tf.device("/device:GPU:0"):
            learner = Learner(sess, *workers[0].get_env_features())
        print("Learner initialized !\n")
        if settings.LOAD:
            learner.load()

        threads = []
        for i in range(settings.NB_ACTORS):
            thread = threading.Thread(target=workers[i].run)
            threads.append(thread)

        threads.append(threading.Thread(target=learner.run))

        if settings.GUI:
            GUI_thread = threading.Thread(target=GUI.main)
            GUI_thread.start()

        sess.run(tf.global_variables_initializer())

        for t in threads:
            t.start()
        print("Running...")

        try:
            while not Actor.STOP_REQUESTED:
                time.sleep(1)
        except KeyboardInterrupt:
            Actor.request_stop()

        for t in threads:
            t.join()

        learner.save()

        DISPLAYER.disp()
        DISPLAYER.disp_q()
        f_t = timeline.Timeline(meta.step_stats)
        chrome_trace = f_t.generate_chrome_trace_format()
        with open("timeline.json", 'w') as f:
            f.write(chrome_trace)

        if settings.GUI:
            GUI_thread.join()
