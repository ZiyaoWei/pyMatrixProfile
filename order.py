import random

class Order:
    def next(self):
        raise NotImplementedError("next() not implemented")


class LinearOrder(Order):
    def __init__(self, m):
        self.m = m
        self.idx = -1

    def next(self):
        self.idx += 1
        if self.idx < self.m:
            return self.idx
        else:
            return None


class RandomOrder(Order):
    def __init__(self, m):
        self.idx = -1
        self.indices = range(m)
        random.shuffle(self.indices)

    def next(self):
        self.idx += 1
        try:
            return self.indices[self.idx]
        except IndexError:
            return None
