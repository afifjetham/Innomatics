import numpy as np

N, M, P = map(int, input().split())
n = np.array([input().split() for i in range(N)], int)
m = np.array([input().split() for i in range(M)], int)
print (np.concatenate((n, m), axis = 0))