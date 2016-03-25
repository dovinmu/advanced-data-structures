'''
Concepts:
-universal hashing
-totally random hashing
-k-wise independence
-chaining: when we get a collision, we make a chain to store both.
-Expected length of chain at slot t = E[Ct] = sum(Pr{ h(xi)=t }). As long as h is universal, for a given xi Pr{ h(xi)=t } = O(1/m), so the total = O(n/m) = O(1) assuming m=Theta(n). This is a very weak bound.
-High-probability bound: If the hashing function is totally random, we expect the len(chain_t) = O(logn/loglogn) w.h.p. (with high probability). So some chains are long.
-But if we look at it amortized over logn accesses, then with high probability we will be taking constant time per access.
-perfect hashing (FKS hashing): store Chain_t = Ct as a hash table of size O(Ct**2). If you get a collision, you rebuild the second hash table.
-Linear probing: try slot h(x), if that's full try slot h(x) + 1. Actually works really well assuming totally random hashing, if you set the size of the table big enough. 5-wise indep. impliest constant overhead, but lower gets pretty bad. Q: seems like it would always be strictly worse than chaining, can it be better?

'''

class Hash():
    def simpleTabulation():
        '''
        -view x as a vector x1, ..., xc of chars
        -totally random hash table T for each char of x
        -Every of c table has size U**1/c for a total of U**epsilon space
        -T1(x1) XOR T2(x2) ... Tc(xc)
        -O(c) time
        -3-wise independent, which is "almost" as good as log-wise independence
        '''
        pass

class BloomFilter():
    pass
