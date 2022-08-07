import numpy as np
np.set_printoptions(legacy = '1.13')

A = np.array(input().split(), float)

print(np.floor(A), np.ceil(A), np.rint(A), sep = '\n')