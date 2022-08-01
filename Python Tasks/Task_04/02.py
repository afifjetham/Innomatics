def split_and_join(line):
    # write your code here
    split = line.split(" ")
    join = "-".join(split)
    return join

line = input()
result = split_and_join(line)
print(result)