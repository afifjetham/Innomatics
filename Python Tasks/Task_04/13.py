def minion_game(string):
    # your code goes here
    s=len(string)
    kevin = 0
    stuart = 0
     
    for i in range(s):
        if string[i] in 'AEIOU':
           kevin+=(s-i)
        else:
           stuart+=(s-i)
                
    if kevin < stuart:
        print('Stuart ' + str(stuart))
    elif kevin > stuart:
        print('Kevin ' + str(kevin))
    else:
        print('Draw')
        
s = input()
minion_game(s)