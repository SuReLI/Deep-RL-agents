
import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DCGAN:

    def __init__(self, sess, learning_rate=2e-4, batch_size=64, nb_gpu=1):

        self.sess = sess

        self.D_learning_rate = 2e-4
        self.G_learning_rate = 2e-4
        self.batch_size = batch_size
        self.nb_gpu = nb_gpu
        self.real_batch_size = batch_size * nb_gpu
        self.fixed_noise = np.random.uniform(-1, 1, (batch_size, 1, 1, 100))

        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        self.build_towers()

        self.writer = tf.summary.FileWriter("./logs/run1", self.sess.graph)

    def generator(self, z, reuse=False):

        with tf.variable_scope('generator', reuse=reuse):

            conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid')
            conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))

            conv2 = tf.layers.conv2d_transpose(conv1, 512, [4, 4], strides=(2, 2), padding='same')
            conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))

            conv3 = tf.layers.conv2d_transpose(conv2, 256, [4, 4], strides=(2, 2), padding='same')
            conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))

            conv4 = tf.layers.conv2d_transpose(conv3, 128, [4, 4], strides=(2, 2), padding='same')
            conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))

            conv5 = tf.layers.conv2d_transpose(conv4, 1, [4, 4], strides=(2, 2), padding='same')
            output = tf.nn.tanh(conv5)

            return output

    def discriminator(self, x, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tf.layers.conv2d(conv1, 256, [4, 4], strides=(2, 2), padding='same')
            conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2))

            conv3 = tf.layers.conv2d(conv2, 512, [4, 4], strides=(2, 2), padding='same')
            conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3))

            conv4 = tf.layers.conv2d(conv3, 1024, [4, 4], strides=(2, 2), padding='same')
            conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4))

            conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(1, 1), padding='valid')
            output = tf.nn.sigmoid(conv5)

            return output, conv5

    def build_tower(self, gpu, x, z):

        G = self.generator(z)
        D_real, D_real_logits = self.discriminator(x)
        D_fake, D_fake_logits = self.discriminator(G, reuse=True)

        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                    labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                    labels=tf.zeros_like(D_fake)))
        D_loss = D_real_loss + D_fake_loss
        
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                    labels=tf.ones_like(D_fake)))

        D_vars = tf.trainable_variables(scope='discriminator')
        G_vars = tf.trainable_variables(scope='generator')

        self.generators.append(G)

        with tf.device('/cpu:0'):
            D_real_loss_summary = tf.summary.scalar(f"D_losses_{gpu}/D_real_loss", D_real_loss)
            D_fake_loss_summary = tf.summary.scalar(f"D_losses_{gpu}/D_fake_loss", D_fake_loss)
            D_loss_summary = tf.summary.scalar(f"D_losses_{gpu}/D_loss", D_loss)
            G_loss_summary = tf.summary.scalar(f"G_loss_{gpu}", G_loss)

            D_real_summary = tf.summary.histogram(f"D_real_{gpu}", D_real)
            D_fake_summary = tf.summary.histogram(f"D_fake_{gpu}", D_fake)
            G_summary = tf.summary.image(f"G_{gpu}", G)

            summary_discriminator = tf.summary.merge([D_real_summary,
                D_real_loss_summary, D_loss_summary])
            summary_generator = tf.summary.merge([D_fake_summary,
                G_summary, D_fake_loss_summary, G_loss_summary])
            summary = tf.summary.merge([summary_discriminator, summary_generator])

        return D_loss, G_loss, D_vars, G_vars, summary

    def average_gradients(self, tower_grads):

        avg_grad = []
        for grad_and_vars in zip(*tower_grads):

            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            avg_grad.append(grad_and_var)

        return avg_grad        

    def build_towers(self):

        D_optimizer = tf.train.AdamOptimizer(self.D_learning_rate, beta1=0.5)
        G_optimizer = tf.train.AdamOptimizer(self.G_learning_rate, beta1=0.5)

        D_grads = []
        G_grads = []
        summaries = []

        self.inputs = tf.placeholder(tf.float32, [self.real_batch_size, 64, 64, 1], name='X')

        shape = (self.batch_size, 1, 1, 100)
        self.z = tf.random_uniform(shape, -1, 1)
        self.z = tf.placeholder_with_default(self.z, shape)

        self.generators = []

        for i in range(self.nb_gpu):

            with tf.device(f'/gpu:{i}'), tf.name_scope(f'Tower_{i}') as scope:

                next_batch = self.inputs[i*self.batch_size:(i+1)*self.batch_size]

                D_loss, G_loss, D_vars, G_vars, summary = self.build_tower(i, next_batch, self.z)

                tf.get_variable_scope().reuse_variables()

                D_grads.append(D_optimizer.compute_gradients(D_loss, var_list=D_vars))
                G_grads.append(G_optimizer.compute_gradients(G_loss, var_list=G_vars))

                summaries.append(summary)

        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):

            D_avg_grad = self.average_gradients(D_grads)
            G_avg_grad = self.average_gradients(G_grads)

            self.D_train = D_optimizer.apply_gradients(D_avg_grad, global_step=self.global_step)
            self.G_train = G_optimizer.apply_gradients(G_avg_grad, global_step=self.global_step)

            self.summary = tf.summary.merge(summaries)

    def generate_image(self, i):
        image = self.sess.run(self.generators[0], feed_dict={self.z: self.fixed_noise})

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

        if not os.path.exists('./results'):
            os.mkdir('./results')

        start_time = time.time()
        for epoch in range(1, epochs+1):
            print(f"Epoch : {epoch} / {epochs}")
            epoch_time = time.time()
            avg_time = []

            nb_batch = int(X.shape[0] / self.real_batch_size)
            for i in range(nb_batch):
                time_i = time.time()

                batch = X[i*self.real_batch_size:(i+1)*self.real_batch_size]

                _, _, summary = self.sess.run([self.D_train, self.G_train, self.summary],
                                              feed_dict={self.inputs: batch})

                self.writer.add_summary(summary, (epoch-1) * nb_batch + i)

                avg_time.append(time.time() - time_i)
                if i % 25 == 0:
                    print(f"\tBatch {i} / {nb_batch}, time", " {:.3f}s".format(avg_time[-1]))

            print("Average time for a batch : {:.3f}s".format(sum(avg_time) / len(avg_time)))
            epoch_time = time.time() - epoch_time
            print("Epoch completed in %.2fs" % epoch_time)
            print("Estimated remaining time : %.3fs" % ((epochs-epoch)*epoch_time), end='\n\n')

            self.generate_image(epoch)

        print("Training completed !")
        print("Total time : %.2f" % (time.time() - start_time))
