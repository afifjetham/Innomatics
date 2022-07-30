happiness = 0
n, m = map(int, input().strip().split(' '))
arr = list(map(int, input().strip().split(' ')))
    
A = set(map(int, input().strip().split(' ')))
B = set(map(int, input().strip().split(' ')))
    
for i in arr:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1
print(happiness)