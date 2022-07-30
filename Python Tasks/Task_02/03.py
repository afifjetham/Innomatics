Result =[]
scorelist = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    Result+=[[name,score]]
    scorelist+=[score]
second_lowest =sorted(set(scorelist))[1] 
for name,score in sorted(Result):
    if score==second_lowest:
        print(name)