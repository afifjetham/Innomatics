M = int(input())
a = set(input().split())
N = int(input())
b = set(input().split())

list1 = list(map(int,a.difference(b)))
list2 = list(map(int,b.difference(a)))

new = list1 + list2

sorted_list = sorted(new)

for i in sorted_list:
    print(i)