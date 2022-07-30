n = int(input())
arr = list(map(int, input().split()))

print(sorted(list(set(arr)))[-2])