
import tensorflow as tf
import numpy as np
from Environment import Environment
from parameters import ENV


class Agent:

	def __init__(self, worker_index, render=False):

		self.worker_index = worker_index
		self.env = Environment()