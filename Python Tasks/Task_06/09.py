import numpy as np

n, m = map(int, input().split())
A = np.array([input().split() for i in range(n)],int)
print(np.prod(np.sum(A, axis=0), axis=0))