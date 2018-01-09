
import numpy as np
import random
from collections import deque

import settings


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=settings.BUFFER_SIZE)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch_size = min(settings.BATCH_SIZE, len(self.buffer))
        return random.sample(self.buffer, batch_size)
