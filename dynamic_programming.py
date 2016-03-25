'''
Following MIT Opencourseware's Intro to Algorithms course taught by Erik Demaine. The code is written to be close to the examples rather than to follow best coding practices.

Five steps to dynamic programming:
1. define subproblems
2. guess part of the solution
3. relate subproblems (recurrance)
4. recurse & memoize, or bottom-up
5. solve original problem
'''

import math

def bellmanFord():
    pass

def textJustification(words, total_width=50):
    '''
    input: list of strings, optional line width int
    '''
    parent = {}
    memo = {}
    #we want to minimize the sum of the badness() of lines
    n = len(words)
    def badness(i, j):
        '''score how well words[i:j] fit into a line.'''
        #TODO: find if the word concat is a serious performance hit
        line = ' '.join(words[i:j])
        if len(line) > total_width:
            return math.inf
        return (total_width - len(line))**3
    def DP(i):
        if i == n: return 0
        if i in memo: return memo[i]
        min_badness = math.inf
        for j in range(i+1, n+1):
            bad = DP(j) + badness(i, j)
            if bad < min_badness:
                min_badness = bad
                parent[i] = j
        memo[i] = min_badness
        return min_badness
    print(DP(0))
    point = 0
    while point in parent:
        print(' '.join(words[point:parent[point]]))
        point = parent[point]

def perfectBlackjack(deck):
    raise NotImplementedError()
    
    memo = {}
    def DP(i):
        if len(deck[i:]) < 4: return 0
        if i in memo: return memo[i]
        for hits in range(0, len(deck)):
            #if over 21, break
            pass
            #DP(i) = max( outcome from {-1,0,1} + DP(i + 4 + #hits + #dealer hits) for #hits in range(0, n) if valid play )
        pass
    pass

def parenthesization():
    pass

def editDistance():
    pass

def knapsack():
    pass

def guitarFingering():
    pass

def tetris():
    pass
