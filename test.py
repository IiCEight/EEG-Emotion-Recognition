

import numpy as np


print([[] for _ in range(3)])

saber = 2

mask = [True] * saber + [False] * (saber + 5)
print(~np.array(mask))