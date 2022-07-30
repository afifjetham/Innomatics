A = set(input().split())
COUNT = 0
VALUE = 0
for i in range(int(input())):
    n = set(input().split())
    if A == n:
        VALUE += 1
    elif A.issuperset(n):
            COUNT += 1
    else:
        VALUE += 1
if VALUE != 0:
    print('False')
else:
    print('True')