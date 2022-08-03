import re
for i in range(int(input())):
    n = (input())
    pattern = re.compile('^[-+]?\d*\.\d+$')
    
    if pattern.search(n):
        print(True)
    else:
        print(False)