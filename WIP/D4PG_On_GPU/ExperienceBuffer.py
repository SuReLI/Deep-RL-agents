
import random
from collections import deque

import settings


class ExperienceBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=settings.MEMORY_SIZE)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self):
        size = min(settings.BATCH_SIZE, len(self.buffer))
        return random.sample(self.buffer, size)

BUFFER = ExperienceBuffer()


class StatsBuffer(ExperienceBuffer):

    def __init__(self):
        super().__init__()
        self.processed = deque(maxlen=settings.MEMORY_SIZE)
        self.non_zeros = 0
        self.removed = []

    def add(self, s, a, r, s_, d):
        super().add(s, a, r, s_, d)

        if len(self.buffer) == settings.MEMORY_SIZE:
            self.non_zeros -= 1
            self.removed.append(self.processed[0])
        self.processed.append(0)

    def sample(self):
        size = min(settings.BATCH_SIZE, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), size)
        for i in idx:
            if self.processed[i] == 0:
                self.non_zeros += 1
            self.processed[i] += 1
        # print("Sampling : ")
        # print(idx)
        # print(self.buffer)
        # print(self.processed)
        # print()
        return [self.buffer[i] for i in idx]

    def stats(self):
        percent = self.non_zeros / len(self.buffer) * 100
        print("Percent seen : %0.3f" % percent)
        
        avg = sum([elt for elt in self.processed]) / self.non_zeros
        print("#seen avg : %0.3f" % avg)
        print("Zeros : ", len(self.buffer) - self.non_zeros)
        try:
            i = self.processed.index(0)
            print("Buffer size : ", len(self.buffer))
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

BUFFER = StatsBuffer()
