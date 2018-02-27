
import random
from collections import deque

from settings import Settings


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=Settings.BUFFER_SIZE)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self):
        size = min(Settings.BATCH_SIZE, len(self.buffer))
        return random.sample(self.buffer, size)
