import re
lst =  list()
for _ in range(int(input())):
    pattern = r"[456]\d{3}(-?\d{4}){3}$"
    s = input()
    if bool(re.match(pattern, s)) is True:
        if bool(re.search(r"([\d])\1\1\1", s.replace("-", ""))) is False:
            print("Valid")
        else:
            print("Invalid")
    else:
        print("Invalid")