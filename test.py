

import numpy as np
import torch
import awkward as ak


# print([[] for _ in range(3)])

adj = [[[1, 0],
       [0, 1]],
       
       [[2, 0],
       [0, 2]]]

x = [[[1, 2],
     [3, 4]],
     [[2, 3],
     [4, 5]]]

adj = torch.Tensor(adj)
x = torch.Tensor(x)

# print(torch.matmul(adj, x))

saber = 2

mask = [True] * saber + [False] * (saber + 5)
# print(~np.array(mask))


x = [[1, 2], [3, 4], [3], [4, 5, 6], [7, 8, 9, 10]]


x = ak.Array(x)

mx = ak.max(x)
mi = ak.min(x)

print("max:", mx)
print("min:", mi)

print(x - mi)