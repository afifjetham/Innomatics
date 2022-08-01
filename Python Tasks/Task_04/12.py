def solve(s):
    for x in s[:].split():
        s = s.replace(x, x.capitalize())
    return s

fptr = open(os.environ['OUTPUT_PATH'], 'w')

s = input()

result = solve(s)

fptr.write(result + '\n')

fptr.close()