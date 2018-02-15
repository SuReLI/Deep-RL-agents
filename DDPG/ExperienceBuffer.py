
import numpy as np
import random
from collections import deque
from settings import Settings


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=Settings.BUFFER_SIZE)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch_size = min(Settings.BATCH_SIZE, len(self.buffer))
        return random.sample(self.buffer, batch_size)
