K = int(input())
l = input().split()
s = set(l)

for ele in list(s):
    l.remove(ele)
    
captain_room = s.difference(set(l)).pop()
print(captain_room)