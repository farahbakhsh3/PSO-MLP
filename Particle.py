import numpy as np


class Particle:
    def __init__(self, data, layers, c1, c2, w, alpha):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.loss = 1
        self.data = data
        self.dspace = sum([layers[i] * layers[i + 1] for i in range(len(layers) - 1)]) + sum(layers)
        self.layers = layers
        self.weights = np.random.normal(*alpha, size=(self.dspace,))
        self.velocity = np.random.normal(*alpha, size=(self.dspace,))
        self.local_best = np.array(self.weights)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def mse(self, x, y):
        pred = self.predict(x)
        mse = sum([(a - b[0]) ** 2 for a, b in zip(y, pred)]) / len(y)
        return mse

    def predict(self, x):
        l = 0
        for i in range(len(self.layers) - 1):
            r, c = self.layers[i], self.layers[i + 1]
            size = r * c
            w = self.weights[l: l + size].reshape((r, c))
            l = l + size
            b = self.weights[l: l + c].reshape((c,))
            l = l + c
            x = self.sigmoid(x.dot(w) + b)

        return x

    def get_loss(self, recalculate=False):
        if recalculate or self.loss is None:
            self.loss = self.mse(*self.data)
        return self.loss

    def __gt__(self, other):
        return self.get_loss() > other.get_loss()

    def __lt__(self, other):
        return self.get_loss() < other.get_loss()

    def __eq__(self, other):
        return self.get_loss() == other.get_loss()

    def update(self, gbest):
        r1, r2 = np.random.rand(2)
        self.velocity = (self.velocity * self.w +
                         self.c1 * r1 * (self.local_best - self.weights) +
                         self.c2 * r2 * (gbest.weights - self.weights))
        self.weights += self.velocity
        loss = self.loss
        if self.get_loss(recalculate=True) < loss:
            self.local_best = np.array(self.weights)

        return self
