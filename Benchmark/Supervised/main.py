
import numpy as np
import tensorflow as tf

from Model import DCGAN
from utils import load_mnist

if __name__ == '__main__':
	
	sess = tf.InteractiveSession()

	X = load_mnist("./data")

	GAN = DCGAN(sess, learning_rate=2e-4, batch_size=100)

	sess.run(tf.global_variables_initializer())

	try:
		GAN.train(X, 2)
	except KeyboardInterrupt:
		pass
