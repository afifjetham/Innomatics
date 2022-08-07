import numpy as np

n = int(input())
a = np.array([input().split() for i in range(n)], int)
b = np.array([input().split() for i in range(n)], int)

print(np.dot(a, b))