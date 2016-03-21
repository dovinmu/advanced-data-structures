import time
import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.style.use('fivethirtyeight')

class TreeNode(object):
    def __init__(self, key, val=True, children=[], parent=None, data=None):
        self.key = key
        self.val = val
        self.children = children
        self.parent = parent
        self.data = data

    def children(self):
        return self.children

class BinaryTreeNode(TreeNode):
    def __init__(self, key, val=True, left=None, right=None, parent=None, data=None):
        super.__init__(key, val, parent=parent, data=data)
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def setLeft(self, node):
        if self.left and self.left.parent is self:
            self.left.parent = None
        self.left = node
        if node:
            node.parent = self

    def setRight(self, node):
        if self.right and self.right.parent is self:
            self.right.parent = None
        self.right = node
        if node:
            node.parent = self

    def isLeft(self, node):
        return node and node.left is self

    def isRight(self, node):
        return node and node.right is self

    def rotate(self):
        if self.parent is None:
            print("cannot rotate on a root!")
            return
        if self is self.parent.left:
            other = self.parent
            new_parent = self.parent.parent
            if new_parent:
                if new_parent.left is other:
                    new_parent.left = self
                else:
                    new_parent.right = self
                self.parent = new_parent
            other.setLeft(self.right)
            self.right = other
            other.parent = self
        elif self is self.parent.right:
            other = self.parent
            new_parent = self.parent.parent
            if new_parent:
                if new_parent.left is other:
                    new_parent.left = self
                else:
                    new_parent.right = self
                self.parent = new_parent
            else:
                self.parent = None
            other.setRight(self.left)
            self.left = other
            other.parent = self
        else:
            print("Cannot find self in parent!")


    def __repr__(self, level=0):
        #TODO: awkward, fix this
        if self.val is not True and self.val is not None:
            return str('{}:{}'.format(self.key, self.val))
        return str(self.key)
        '''Print tree structure (needs work)
        left = self.left.__repr__(level) if self.left else ''
        right = self.right.__repr__(level+1) if self.right else ''
        left = left.split('\n')
        right = right.split('\n')
        combined = ''
        while left or right:
            if left:
                combined += left.pop(0) + ' '
            else:
                combined += ' ' * level
            if right:
                combined += right.pop(0) + ' '
            combined += '\n'
        return str('{0}\n|\\\n{1}'.format(self.key, combined))
        '''

class Tree(object):
    def __init__(self):
        self.root = None

    def height(self, node=None):
        if not node:
            node = self.root
        leaves = [node]
        level = 0
        while len(leaves) > 0:
            new_leaves = []
            for leaf in leaves:
                if leaf:
                    for child in leaf.children():
                        if child:
                            new_leaves.append(child)
            level += 1
            leaves = new_leaves
        return level-1

    def getLevel(self, lvl):
        leaves = [self.root]
        level = 0
        while level < lvl and leaves:
            new_leaves = []
            for leaf in leaves:
                if leaf:
                    for child in leaf.children():
                        if child:
                            new_leaves.append(child)
            level += 1
            leaves = new_leaves
        return leaves

    def checkTreeProperty(self):
        seen = {}
        leaves = [self.root]
        while leaves:
            new_leaves = []
            for leaf in leaves:
                for child in leaf.children():
                    if child:
                        if child in seen:
                            print ('Multiple parents: {0}->{1}'.format(leaf.key, child.key))
                            return False
                        else:
                            new_leaves.append(child)
                        if leaf.left.parent != leaf:
                            print ('erroneous back-pointer:{0}->{1}'.format(leaf.key, child.key))
                            return False
                seen[leaf] = True
            leaves = new_leaves
        print("Looks like a tree.")
        return True

class BinarySearchTree(Tree):
    def check(self):
        return self.checkTreeProperty() and self.checkBinarySearchProperty()

    def checkBinarySearchProperty(self):
        leaves = [self.root]
        while leaves:
            new_leaves = []
            for leaf in leaves:
                if leaf.right:
                    if leaf.right.key < leaf.key:
                        print('Nope: {} < {} is wrong'.format(leaf.right, leaf))
                        return False
                        new_leaves.append(leaf.right)
                        if leaf.left:
                            if leaf.left.key > leaf.key:
                                print("Nope: {} > {} is wrong.".format(leaf.left, leaf))
                                return False
                                new_leaves.append(leaf.left)
                                leaves = new_leaves
                                print("This tree is searchable, yay!")
                                return True

    def depth(self, key):
        node = BinarySearchTree.search(self, key)
        if not node:
            return -1
        ret = 0
        while node.parent:
            node = node.parent
            ret += 1
        return ret

    def insert(self, key, val=True):
        if self.root is None:
            self.root = BinaryTreeNode(key, val)
            return self.root
        else:
            return self.insertRecursively(key, val, self.root)

    def insertRecursively(self, key, val, node):
        if key > node.key:
            if node.right is None:
                node.right = BinaryTreeNode(key, val, parent=node)
                return node.right
            else:
                return self.insertRecursively(key, val, node.right)
        else:
            if node.key == key:
                print("Attempting to insert a duplicate... disallowed")
                return None
            if node.left is None:
                node.left = BinaryTreeNode(key, val, parent=node)
                return node.left
            else:
                return self.insertRecursively(key, val, node.left)

    def search(self, key):
        if self.root is None:
            return None
        node = self.searchRecursively(key, self.root)
        #print('got', node)
        return node

    def searchRecursively(self, key, node):
        #print('hit key ', node.key)
        if key == node.key:
            return node
        if key > node.key and node.right:
            return self.searchRecursively(key, node.right)
        elif key < node.key and node.left:
            return self.searchRecursively(key, node.left)
        return None

    def rotate(self, key):
        node = self.search(key)
        node.rotate()
        if node.parent is None:
            self.root = node

    def __repr__(self):
        if self.root:
            return self.stringify(self.root)

    def stringify(self, node, level=0):
        ret = '. ' * level+repr(node)+"\n"
        if node.left:
            ret += self.stringify(node.left, level+1)
        else:
            ret += '. '*(level+1) + '#' + '\n'
        if node.right:
            ret += self.stringify(node.right, level+1)
        else:
            ret += '. '*(level+1) + '#' + '\n'

        return ret

#challenge: given x, cut a tree into two red-black trees with one
#tree's values < x and one trees's values > x
class RedBlackTree(BinarySearchTree):
    def insert(self, key, val=True):
        node = BinarySearchTree.insert(self, key, val)
        #how do I know what to color the node?


class SplayTree(BinarySearchTree):
    '''In a splay tree, accessing a node x of a BST brings it to the root.
       Thus, recently accessed items are quick to access again.
       Insertion, look-up and removal are O(logn) amortized. For sequences
       of non-random operations, splay trees perform better than normal BSTs.

       Conjectured to be dynamically optimal
    '''
    def search(self, key):
        node = BinarySearchTree.search(self, key)
        self.splay(node)
        return node

    def splay(self, node):
        while node.parent:
            #print('height:', self.height(node))
            grandparent = node.parent.parent
            if not grandparent:
#                print('zig', end=' ')
                node.rotate()
            elif node.isLeft(node.parent) and node.parent.isLeft(grandparent):
                #print('zig-zig', end=' ')
                #node.parent.rotate()
                #node.rotate()
                B = node.right
                C = node.parent.right
                node.setRight(node.parent)
                if grandparent.parent:
                    if grandparent.isLeft(grandparent.parent):
                        grandparent.parent.setLeft(node)
                    else:
                        grandparent.parent.setRight(node)
                else:
                    node.parent = None
                node.right.setLeft(B)
                node.right.setRight(grandparent)
                grandparent.setLeft(C)
            elif node.isRight(node.parent) and node.parent.isRight(grandparent):
#                print('zig-zig', end=' ')
                #node.parent.rotate()
                #node.rotate()
                B = node.parent.left
                C = node.left
                node.setLeft(node.parent)
                if grandparent.parent:
                    if grandparent.isLeft(grandparent.parent):
                        grandparent.parent.setLeft(node)
                    else:
                        grandparent.parent.setRight(node)
                else:
                    node.parent = None
                node.left.setRight(C)
                node.left.setLeft(grandparent)
                grandparent.setRight(B)
            else:
#                print('zig-zag', end=' ')
                node.rotate()
                node.rotate()
        self.root = node

class RangeTree(BinarySearchTree):
    '''
    A range tree on a set of 1-dimensional points is a balanced binary search tree on those points. The points stored in the tree are stored in the leaves of the tree; each internal node stores the largest value contained in its left subtree. A range tree on a set of points in d-dimensions is a recursively defined multi-level binary search tree. Each level of the data structure is a binary search tree on one of the d-dimensions. The first level is a binary search tree on the first of the d-coordinates. Each vertex v of this tree contains an associated structure that is a (d−1)-dimensional range tree on the last (d−1)-coordinates of the points stored in the subtree of v.
    '''
    def __init__(d):
        raise NotImplementedError()


class TangoTree(BinarySearchTree):
    '''
    The basic idea of a Tango Tree is to store "preferred child" paths,
    where each node stores its most recently accessed child as the
    preferred child. These paths can be stored as auxiliary BBSTs
    sorted by the original keys.
    '''
    def __init__(d):
        raise NotImplementedError()


'''
___Various properties to care about___
Sequential Access property:
Dynamic Finger:
Working set property:
Unified property: attempts to unify Dynamic Finger and Working Set.
   if t_ij distinct keys accessed in x_i, ..., x_j then x_j costs
   O(lg min_i [|x_i - x_j| + t_ij + 2])
   What we want to find is to find a key that minimizes both being
   in the recent past and close in space. Think about it like growing
   a box outwards from the staring location in space and time.
   This can be achieved with a pointer machine DS, but we don't know
   how to do it with a BST.
Dynamic Optimality (aka O(1)-competitive): we want for the total cost
   to be within the same order as the optimal solution, i.e. that if the
   cost is O(1) when we know the future then the cost is O(1) when we don't. All BBSTs are O(logn)-competitive, and we know how to do
   O(loglogn). Splay trees might be dynamically optimal, but we don't know.
'''

def treeCompare(load_seq, access_seq):
    tree_dict = {'BST':BinarySearchTree(), 'SplayTree':SplayTree()}
    for name,tree in tree_dict.items():
        for item in load_seq:
            tree.insert(item)
    for name,tree in tree_dict.items():
        t0 = time.time()
        for item in access_seq:
            tree.search(item)
        t1 = time.time()
        print("{0} took {1} seconds. Height: {2}".format(name, int((t1-t0)*1000)/1000, tree.height()))

def treeRace():
    #TODO: compare the speed of each tree for lots of access sequences on the given data
    '''
    performance for trees with randomly ordered ranges from 0 to n-1, using
    a subset of size n/m:
    TODO: fill these in
        full set:
        (n = 1000, m = 1, size=1000)
        (n = 10000, m = 1, size=10000)
        1/10 subset:
        (n = 1000, m = 10, size=100)
        (n = 10000, m = 10, size=1000)
        1/100 subset:
        (n = 1000, m = 100, size=10)
        (n = 10000, m = 100, size=100)
        1/1000 subset:
        (n = 10000, m = 1000, size=10)

    Then: fill in for range(1,k) where k < m
    '''
    n = 100000
    m = 100

    bst = BinarySearchTree()
    splay = SplayTree()

    seq = [i for i in range(n)]
    np.random.shuffle(seq)
    print('inerting {} values'.format(n))
    for num in seq:
        bst.insert(num)
        splay.insert(num)
    print(bst)
    print('Racing Splay Tree vs Binary Search Tree (unbalanced, but randomly built) on access times for a reapeating subset 0 <= num < {} of {}k total inserted elements.'.format(m, int(n/100)/10))
    np.random.shuffle(seq)
    splay_series = []
    bst_series = []
    #subset = seq[:int(n/m)]
    subset = [num for num in seq if num < m]
    print('subset:', subset)
    t0 = time.time()
    for i in range(10):
        np.random.shuffle(subset)
        for num in subset:
            bst_depth = bst.depth(num)
            bst_series.append(bst_depth)
            bst.search(num)
    t1 = time.time()
    for i in range(10):
        np.random.shuffle(subset)
        for num in subset:
            splay_depth = splay.depth(num)
            splay_series.append(splay_depth)
            splay.search(num)
    t2 = time.time()

    splay_series = Series(splay_series)
    bst_series = Series(bst_series)

    bst_series.plot(alpha=0.2)
    splay_series.plot(alpha=0.2)

    bst_series.ewm(span=int(n/m)).mean().plot(style='g--')
    splay_series.ewm(span=int(n/m)).mean().plot(style='k--')

    plt.title("BST time: {0}  Splay tree time: {1}".format(int((t1-t0)*1000)/1000,int((t2-t1)*1000)/1000))
    plt.show()
    '''
    uniform_rand = [random.random() for i in range(n)]
    load = np.copy(uniform_rand)
    print("{} uniform random nums, sorted access".format(n))
    treeCompare(load, sorted(uniform_rand))
    print("{} uniform random nums, shuffled access".format(n))
    np.random.shuffle(uniform_rand)
    treeCompare(load, uniform_rand)

    #This should be what a splay tree is the best at!
    uniform_rand = [num for num in uniform_rand if num < 0.2]
    print("{} uniform random numbers, random load / subset access (<.2)".format(n))
    treeCompare(load, uniform_rand * 5)
    '''

def compareDepthAccessTimes(n = 10000):
    '''Written in an effort to optimize Splay tree, which seems to be losing to
    BST more often than not even in non-random access sequences'''
    bst = BinarySearchTree()
    splay = SplayTree()

    seq = [i for i in range(n)]
    np.random.shuffle(seq)
    print('inerting {} values'.format(n))
    for num in seq:
        bst.insert(num)
        splay.insert(num)
    print(bst)
    bst_series = []
    splay_series = []
    k = min(bst.height(), splay.height())
    print('timing access from depth 0 to depth {}'.format(k))
    for i in range(k):
        bst_key = bst.getLevel(i)[0].key
        level = splay.getLevel(i)
        if level:
            splay_key = level[0].key
        else:
            break
        t0 = time.time()
        bst.search(bst_key)
        t1 = time.time()
        splay.search(splay_key)
        t2 = time.time()
        bst_series.append((t1-t0)*1000)
        splay_series.append((t2-t1)*1000)
    df = DataFrame()
    df['BST'] = bst_series
    df['Splay'] = splay_series
    df.plot(title="BST and Splay access time by depth for trees of {} items".format(n))
    plt.xlabel('depth')
    plt.ylabel('ms')
    plt.show()


'''
#interesting attempt to print tree structure
#check out http://blog.mikedll.com/2011/04/red-black-trees-in-python.html
#for possible awesome console based printing of trees
'''
