
import numpy as np
import random
from collections import deque

import settings

import matplotlib.pyplot as plt
plt.ion()


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=settings.MEMORY_SIZE)
        self.processed = deque(maxlen=settings.MEMORY_SIZE)
        self.non_zero = 0
        self.removed = []

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) == settings.MEMORY_SIZE:
            self.non_zero -= 1
            self.removed.append(self.processed[0])
        self.buffer.append(experience)
        self.processed.append(0)

    def sample(self):
        size = min(settings.BATCH_SIZE, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), size)
        for i in idx:
            if self.processed[i] == 0:
                self.non_zero += 1
            self.processed[i] += 1
        # print("Sampling : ")
        # print(idx)
        # print(self.buffer)
        # print(self.processed)
        # print()
        return [self.buffer[i] for i in idx]

    def stats(self):
        percent = self.non_zero / len(self.buffer) * 100
        print("Percent seen : %0.3f" % percent)
        
        avg = sum([elt for elt in self.processed]) / self.non_zero
        print("#seen avg : %0.3f" % avg)
        print("Zeros : ", len(self.buffer) - self.non_zero)
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

    def disp(self):
        plt.gcf().clear()
        plt.plot(self.removed)
        plt.show(block=False)
        plt.pause(0.5)
