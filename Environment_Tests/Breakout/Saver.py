# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import time

import settings


class Saver:

    def __init__(self):
        pass

    def set_sess(self, sess):
        self.saver = tf.train.Saver()
        self.sess = sess

    def save(self, global_total_time, wall_time, total_eps, total_steps):

        print("Saving model...")
        os.makedirs(os.path.dirname("model/"), exist_ok=True)

        file_stats = 'model/stats.' + str(total_steps)

        with open(file_stats, 'w') as file:
            file.write(str(global_total_time) + "\n")
            file.write(str(wall_time) + "\n")
            file.write(str(total_eps) + "\n")
            file.write(str(total_steps))

        self.saver.save(self.sess, "model/checkpoint", global_step=total_steps)
        print("Model saved !")

    def load(self):

        if not settings.LOAD:
            return 0, 0, 0, 0

        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state("model/")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

            print("Checkpoint loaded :", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")

            # Load global step
            global_step = int(tokens[1])

            # Load stats
            file_stats = 'model/stats.' + str(global_step)
            with open(file_stats, 'r') as file:
                text = file.read().split("\n")
                global_total_time = float(text[0])
                wall_time = float(text[1])
                total_eps = int(text[2])
                total_steps = int(text[3])

            print("Model loaded ! Resuming session :")
            print("Wall time : %f\nCpu time : %f\n"
                  "Total episodes : %i\nTotal steps %i" %
                  (wall_time, global_total_time, total_eps, total_steps))
            return global_total_time, wall_time, total_eps, total_steps

        else:
            print("Could not find old checkpoint")

            # Reinitialize global_total_time, wall_time, total_eps, total_steps
            return 0, 0, 0, 0

SAVER = Saver()
