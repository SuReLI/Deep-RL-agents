
import numpy as np
import random

import parameters


class ExperienceBuffer:

    def __init__(self):
        self.buffer = []
        self.buffer_size = parameters.BUFFER_SIZE

    def add(self, experiences):
        if len(self.buffer) + len(experiences) > self.buffer_size:
            deb = len(experiences) + len(self.buffer) - self.buffer_size
            del self.buffer[0:deb]
        self.buffer.extend(experiences)

    def sample(self, n):
        samples = np.array(random.sample(self.buffer, n))
        return np.reshape(samples, [n, 5])
