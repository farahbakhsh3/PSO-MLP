import numpy as np
from PSOMLP import PSOMLP


# XOR
x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [0, 1, 1, 0]

x = np.array(x)
y = np.array(y)

pso = PSOMLP(hlayers=(10,))
mlp = pso.fit(x, y, iterations=100, nparticles=100)
print("Loss for trainning data:", mlp.mse(x, y))

for i in x:
    print(i, ' -> ', mlp.predict(np.array(i).reshape(1, -1))[0])

print('=======================')
print(mlp.layers)
print('=======================')

l = 0
for i in range(len(mlp.layers) - 1):
    r, c = mlp.layers[i], mlp.layers[i + 1]
    size = r * c
    w = mlp.weights[l: l + size].reshape((r, c))
    l = l + size
    b = mlp.weights[l: l + c].reshape((c,))
    l = l + c

    print(i)
    print(w)
    print(b)
    print('------------')
