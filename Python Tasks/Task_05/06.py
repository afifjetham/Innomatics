from sys import stdin
import re
n = input()
print(re.sub( r"(?<= )(&&|\|\|)(?= )", lambda x: "and" if x.group()=="&&" else "or", stdin.read()))