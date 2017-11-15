# -*- coding: utf-8 -*-
import tensorflow as tf
import threading

import signal
import math
import time

from Network import Network
from Agent import Agent
from rmsprop_applier import RMSPropApplier

from Displayer import DISPLAYER
from Saver import SAVER

import settings


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def work(worker_index):
    global global_total_time, total_eps, total_steps

    worker = workers[worker_index]

    while not stop_requested and total_steps <= settings.MAX_TIME_STEP:
        elapsed_time, done, steps = worker.process(sess,
                                                   total_steps,
                                                   summary_writer,
                                                   summary_op,
                                                   score_input)
        global_total_time += elapsed_time
        total_eps += done
        total_steps += steps

    worker.close()


def signal_handler(signal, frame):
    global stop_requested

    print('End of training')
    stop_requested = True


if __name__ == '__main__':

    tf.reset_default_graph()

    device = "/cpu:0"

    initial_learning_rates = log_uniform(settings.INITIAL_ALPHA_LOW,
                                         settings.INITIAL_ALPHA_HIGH,
                                         settings.INITIAL_ALPHA_LOG_RATE)
    stop_requested = False

    print("Creating the global network...")
    global_network = Network(0, device)

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                  decay=settings.RMSP_ALPHA,
                                  momentum=0.0,
                                  epsilon=settings.RMSP_EPSILON,
                                  clip_norm=settings.MAX_GRADIENT_NORM,
                                  device=device)
    print("Global network created !")

    # Create and initialize the workers
    workers = []
    for i in range(settings.NB_THREADS):
        print("\nCreating worker %i..." % (i + 1))
        worker = Agent(i + 1,
                       global_network,
                       initial_learning_rates,
                       learning_rate_input,
                       grad_applier,
                       device)

        workers.append(worker)
    print("\nEvery worker has been created !")

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    sess.run(tf.global_variables_initializer())

    SAVER.set_sess(sess)

    # summary for tensorboard
    score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", score_input)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(settings.LOG_FILE, sess.graph)

    # Load checkpoint with saver
    global_total_time, wall_time, total_eps, total_steps = SAVER.load()

    # Create one thread per worker
    train_threads = []
    for i in range(settings.NB_THREADS):
        train_threads.append(threading.Thread(target=work,
                                              args=(i,)))

    # Intercept CTRL+C signal
    signal.signal(signal.SIGINT, signal_handler)

    # Set start time
    global_start_time = time.time()

    # Start the workers' thread
    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')
    signal.pause()


    for t in train_threads:
        t.join()

    wall_time += time.time() - global_start_time

    print("Wall time of simulation : %f\nGlobal CPU time : %f\n"
          "Total number of episodes : %i\nTotal number of steps : %i" %
          (wall_time, global_total_time, total_eps, total_steps))

    print('Now saving data. Please wait')
    SAVER.save(global_total_time, wall_time, total_eps, total_steps)
    DISPLAYER.disp()

    summary_writer.close()
