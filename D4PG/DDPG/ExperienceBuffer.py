
import numpy as np
import random
from collections import deque

import settings


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=settings.MEMORY_SIZE)
        self.processed = deque(maxlen=settings.MEMORY_SIZE)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)
        self.processed.append(0)

    def sample(self):
        size = min(settings.BATCH_SIZE, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), size)
        for i in idx:
            self.processed[i] += 1
        # print("Sampling : ")
        # print(idx)
        # print(self.buffer)
        # print(self.processed)
        # print()
        return [self.buffer[i] for i in idx]

    def stats(self):
        non_zero = sum(map(lambda x:1 if x else 0, self.processed))
        percent = non_zero / len(self.buffer) * 100
        print("Percent seen : %0.3f" % percent)
        
        avg = sum([elt for elt in self.processed]) / non_zero
        print("#seen avg : %0.3f" % avg)
        print("Zeros : ", len(self.buffer) - non_zero)
        try:
            i = self.processed.index(0)
            print(len(self.buffer))
            print(i)
            print(list(self.processed)[i-40:i+40])
        except ValueError:
            pass
        print(list(self.processed)[:100])

        total_avg = sum(self.processed) / len(self.buffer)
        print("#seen total avg : %0.3f" % total_avg)
        print()

