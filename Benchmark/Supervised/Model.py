
import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DCGAN:

    def __init__(self, sess, learning_rate=2e-4, batch_size=64):

        self.sess = sess

        self.lr = learning_rate
        self.batch_size = batch_size
        self.fixed_noise = np.random.uniform(-1, 1, (25, 1, 1, 100))

        self.build_model()

        self.writer = tf.summary.FileWriter("./logs/run1", self.sess.graph)

    def generator(self, z, training=True, reuse=False):

        with tf.variable_scope('generator', reuse=reuse):

            conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid')
            conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=training))

            conv2 = tf.layers.conv2d_transpose(conv1, 512, [4, 4], strides=(2, 2), padding='same')
            conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=training))

            conv3 = tf.layers.conv2d_transpose(conv2, 256, [4, 4], strides=(2, 2), padding='same')
            conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=training))

            conv4 = tf.layers.conv2d_transpose(conv3, 128, [4, 4], strides=(2, 2), padding='same')
            conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=training))

            conv5 = tf.layers.conv2d_transpose(conv4, 1, [4, 4], strides=(2, 2), padding='same')
            output = tf.nn.tanh(conv5)

            return output

    def discriminator(self, x, training=True, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tf.layers.conv2d(conv1, 256, [4, 4], strides=(2, 2), padding='same')
            conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=training))

            conv3 = tf.layers.conv2d(conv2, 512, [4, 4], strides=(2, 2), padding='same')
            conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=training))

            conv4 = tf.layers.conv2d(conv3, 1024, [4, 4], strides=(2, 2), padding='same')
            conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=training))

            conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(1, 1), padding='valid')
            output = tf.nn.sigmoid(conv5)

            return output, conv5

    def build_model(self):

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='x')
        self.z = tf.placeholder(tf.float32, [None, 1, 1, 100], name='z')

        self.training = tf.placeholder(dtype=tf.bool)

        self.G = self.generator(self.z, self.training)
        D_real, D_real_logits = self.discriminator(self.x, self.training)
        D_fake, D_fake_logits = self.discriminator(self.G, self.training, reuse=True)

        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                    labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                    labels=tf.zeros_like(D_fake)))
        self.D_loss = D_real_loss + D_fake_loss
        
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                    labels=tf.ones_like(D_fake)))

        D_vars = tf.trainable_variables(scope='discriminator')
        G_vars = tf.trainable_variables(scope='generator')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            D_trainer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            self.D_optimize = D_trainer.minimize(self.D_loss, var_list=D_vars)

            G_trainer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            self.G_optimize = G_trainer.minimize(self.G_loss, var_list=G_vars)

        self.D_real_loss_summary = tf.summary.scalar("D_losses/D_real_loss", D_real_loss)
        self.D_fake_loss_summary = tf.summary.scalar("D_losses/D_fake_loss", D_fake_loss)
        self.D_loss_summary = tf.summary.scalar("D_losses/D_loss", self.D_loss)
        self.G_loss_summary = tf.summary.scalar("G_loss", self.G_loss)

        self.D_real_summary = tf.summary.histogram("D_real", D_real)
        self.D_fake_summary = tf.summary.histogram("D_fake", D_fake)
        self.G_summary = tf.summary.image("G", self.G)

    def generate_image(self, i):
        feed_dict = {self.z: self.fixed_noise, self.training: False}
        image = self.sess.run(self.G, feed_dict=feed_dict)

        fig, ax = plt.subplots(5, 5, figsize=(5, 5))
        for k in range(25):
            a = ax[k//5, k%5]
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.cla()
            a.imshow(np.reshape(image[k], (64, 64)), cmap='gray')

        fig.text(0.5, 0.04, f"Epoch {i}", ha='center')

        plt.savefig(f"./results/gen_batch_{i}.png")
        plt.close()

    def train(self, X, epochs):
        print("Start training...")

        self.summary_discriminator = tf.summary.merge([self.D_real_summary,
            self.D_real_loss_summary, self.D_loss_summary])
        self.summary_generator = tf.summary.merge([self.D_fake_summary,
            self.G_summary, self.D_fake_loss_summary, self.G_loss_summary])

        os.makedirs("./results", exist_ok=True)
        train_hist = {"D_losses": [], "G_losses": []}
        start_time = time.time()

        for epoch in range(1, epochs+1):
            print(f"Epoch : {epoch} / {epochs}")
            D_losses = []
            G_losses = []
            epoch_time = time.time()
            avg_time = []

            nb_batch = X.shape[0] // self.batch_size
            for i in range(nb_batch):
                time_i = time.time()

                batch = X[i*self.batch_size:(i+1)*self.batch_size]
                z = np.random.uniform(-1, 1, (self.batch_size, 1, 1, 100))

                feed_dict = {self.x: batch, self.z: z, self.training: True}
                D_loss, summary, _ = self.sess.run(
                    [self.D_loss, self.summary_discriminator, self.D_optimize],
                    feed_dict=feed_dict)
                self.writer.add_summary(summary, (epoch-1) * nb_batch + i)

                z = np.random.uniform(-1, 1, (self.batch_size, 1, 1, 100))
                feed_dict = {self.x: batch, self.z: z, self.training: True}
                G_loss, summary, _ = self.sess.run(
                    [self.G_loss, self.summary_generator, self.G_optimize],
                    feed_dict=feed_dict)

                z = np.random.uniform(-1, 1, (self.batch_size, 1, 1, 100))
                feed_dict = {self.x: batch, self.z: z, self.training: True}
                G_loss, summary, _ = self.sess.run(
                    [self.G_loss, self.summary_generator, self.G_optimize],
                    feed_dict=feed_dict)
                self.writer.add_summary(summary, (epoch-1) * nb_batch + i)

                D_losses.append(D_loss)
                G_losses.append(G_loss)

                avg_time.append(time.time() - time_i)
                if i % 25 == 0:
                    print(f"\tBatch {i} / {nb_batch}, time", " {:.3f}s".format(avg_time[-1]))

            train_hist["D_losses"].append(np.mean(D_losses))
            train_hist["G_losses"].append(np.mean(G_losses))

            print("Average time for a batch : {:.3f}s".format(sum(avg_time) / len(avg_time)))
            epoch_time = time.time() - epoch_time
            print("Epoch completed in %.2fs : D loss = %.4f, G loss = %.4f" %
                (epoch_time, np.mean(D_losses), np.mean(G_losses)))
            print("Estimated remaining time : %.3fs" % ((epochs-epoch)*epoch_time), end='\n\n')

            try:
                self.generate_image(epoch)
            except:
                pass

        print("Training completed !")
        print("Total time : %.2f" % (time.time() - start_time))

