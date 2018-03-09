
import numpy as np

class SumTree:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.sum_tree = np.zeros(2 * capacity - 1)
        self.max_tree = np.ones(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, value):
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1

        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        self.max_tree[parent] = max(self.max_tree[left], self.max_tree[right])

        if parent != 0:
            self._propagate(parent, value)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1

        if left >= 2 * self.capacity - 1:
            return idx

        if value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            right = left + 1
            return self._retrieve(right, value - self.sum_tree[left])

    def total(self):
        return self.sum_tree[0]

    def max(self):
        return self.max_tree[0]

    def add(self, value, data):
        self.data[self.write] = data
        self.update(self.write + self.capacity - 1, value)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, value):
        self.sum_tree[idx] = self.max_tree[idx] = value
        self._propagate(idx, value)

    def get(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1

        return self.data[data_idx], idx, self.sum_tree[idx]

    def sample(self, batch_size):
        batch_idx = [None] * batch_size
        batch_priorities = [None] * batch_size
        batch = [None] * batch_size
        segment = self.total() / batch_size

        a = [segment*i for i in range(batch_size)]
        b = [segment * (i+1) for i in range(batch_size)]
        s = np.random.uniform(a, b)

        for i in range(batch_size):
            (batch[i], batch_idx[i], batch_priorities[i]) = self.get(s[i])

        return batch, batch_idx, batch_priorities

    def __repr__(self):
        s = ""
        last_line = " ".join([str(self.sum_tree[i+self.capacity-1]).center(5) for i in range(len(self.data))])
        for i in range(4):
            line = self.sum_tree[2**i-1:2**i-1+2**i]
            s += (" "*(len(last_line)//2**(i+1))).join([str(line[i]).center(5) for i in range(len(line))]).center(len(last_line)) + "\n"
        s += last_line
        return s
