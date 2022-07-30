n = int(input())

int_list = []

for i in input().split():
    int_list.append(int(i))
    
t = tuple(int_list)
print(hash(t))