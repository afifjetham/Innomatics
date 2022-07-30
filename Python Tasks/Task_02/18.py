for i in range(int(input())):
    ele = int(input())
    a = set(map(int, input().split()))

    ele1 = int(input())
    b = set(map(int, input().split()))

    if a.issubset(b):
        print(True)
    else:
        print(False)