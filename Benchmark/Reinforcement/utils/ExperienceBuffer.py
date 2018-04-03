
import random
from collections import deque
import numpy as np

from SumTree import SumTree
from settings import Settings


def ExperienceBuffer(*, prioritized=False):
	if prioritized:
		return PrioritizedExperienceBuffer()
	else:
		return RegularExperienceBuffer()

class RegularExperienceBuffer:

	def __init__(self):
		self.buffer = deque(maxlen=Settings.BUFFER_SIZE)

	def __len__(self):
		return len(self.buffer)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, beta=None):
		batch_size = min(Settings.BATCH_SIZE, len(self.buffer))
		return random.sample(self.buffer, batch_size)

	def update(self, idx, errors):
		pass


class PrioritizedExperienceBuffer:

	def __init__(self):
		self.buffer = SumTree(capacity=Settings.BUFFER_SIZE)

	def add(self, experience):
		self.buffer.add(self.buffer.max(), experience)

	def sample(self, beta):
		data, idx, priorities = self.buffer.sample(Settings.BATCH_SIZE)
		probs = priorities / self.buffer.total()
		weights = (self.buffer.n_entries * probs) ** -beta
		weights /= np.max(weights)

		return data, idx, weights

	def update(self, idx, errors):
		priorities = (np.abs(errors) + 1e-6) ** Settings.ALPHA
		for i in range(len(idx)):
			self.buffer.update(idx[i], priorities[i])
