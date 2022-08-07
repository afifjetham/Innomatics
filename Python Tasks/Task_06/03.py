N,M = list(map( int, input().split()))
arr = [list(map(int, input().split())) for i in range(N)]
np_arr = np.array(arr)
print(np_arr.transpose())
print(np_arr.flatten())