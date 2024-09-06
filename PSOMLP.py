import copy
from tqdm import tqdm
from Particle import Particle


class PSOMLP:
    def __init__(self, hlayers=(100,), c1=2.05, c2=2.05, w=0.9, alpha=(-1, 1)):
        self.gbest = None
        self.swarm = None
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.hlayers = hlayers

    def fit(self, x, y, iterations=100, nparticles=100, print_step=True):
        n_in = 1 if len(x.shape) == 1 else x.shape[1]
        n_out = 1 if len(y.shape) == 1 else y.shape[1]
        layers = [n_in, *self.hlayers, n_out]
        self.swarm = [Particle([x, y], layers, self.c1, self.c2, self.w, self.alpha) for _ in range(nparticles)]
        self.gbest = min(self.swarm)
        loops = tqdm(range(iterations), desc="Trainning MLP") if print_step else range(iterations)
        nbest = copy.deepcopy(self.gbest)
        for _ in loops:
            for p in self.swarm:
                if p.update(self.gbest) < nbest:
                    nbest = copy.deepcopy(p)

            if nbest < self.gbest:
                self.gbest = copy.deepcopy(nbest)

        return self.gbest
