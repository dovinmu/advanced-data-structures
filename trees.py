import time
import numpy as np
import sys
import inspect
import random
import pandas as pd

import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.style.use('fivethirtyeight')

class TreeNode(object):
    def __init__(self, key, val=True, children=[], parent=None, data={}):
        self.key = key
        self.val = val
        self._children = children
        self.parent = parent
        self.data = data

    def children(self):
        return self._children

class BinaryTreeNode(TreeNode):
    def __init__(self, key, val=True, left=None, right=None, parent=None, data={}):
        super().__init__(key, val, parent=parent, data=data)
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
        if self.data.items():
            return str('{}:{}'.format(self.key, self.data))
        return str(self.key)

'''TODO: add these functions to all trees (from https://en.wikipedia.org/wiki/Tree_(data_structure))
Enumerating all the items
Enumerating a section of a tree
Searching for an item
Adding a new item at a certain position on the tree
Deleting an item
Pruning: Removing a whole section of a tree
Grafting: Adding a whole section to a tree
Finding the root for any node
'''
class Tree(object):
    def __init__(self):
        self.root = None

    def depth(self, node):
        if not node:
            return None
        ret = 0
        while node.parent:
            node = node.parent
            ret += 1
        return ret

    def updateHeight(self, node):
        if node:
            left = self.updateHeight(node.left)
            right = self.updateHeight(node.right)
            node.data['height'] = max(left, right)+1
            return node.data['height']
        return -1

    def height(self, node=None):
        if not node:
            return -1
        if 'height' in node.data:
            return node.data['height']
        print('computing height')
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
                        if child.parent != leaf:
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

    def insert(self, key, val=True, data={}):
        if self.root is None:
            self.root = BinaryTreeNode(key, val, data=data)
            return self.root
        else:
            node = self.insertRecursively(key, val, self.root)
            node.data = data
            return node

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



class SplayTree(BinarySearchTree):
    '''In a splay tree, accessing a node x of a BST brings it to the root.
       Thus, recently accessed items are quick to access again.
       Insertion, look-up and removal are O(logn) amortized. For sequences
       of non-random operations, splay trees perform better than normal BSTs.

       Conjectured to be dynamically optimal, but this implementation generally
       performs at least slightly worse than a regular BST.
    '''
    def search(self, key, splay=True):
        node = BinarySearchTree.search(self, key)
        if splay:
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

#challenge: given x, cut a tree into two red-black trees with one
#tree's values < x and one trees's values > x
class RedBlackTree(BinarySearchTree):
    pass

class AVLTree(BinarySearchTree):
    #AVL property: height difference between left and right subtrees no greater than 1
    def insert(self, key, val=True):
        node = BinarySearchTree.insert(self, key, val, data={'height':0})
        self.fixAVL(node)

    def fixAVL(self, node):
        #print(self)
        while node and node.parent:
            self.updateHeight(node.parent)
            parent_balance = self.height(node.parent.left) - self.height(node.parent.right)
            node_balance = self.height(node.left) - self.height(node.right)
            if parent_balance > 1: #left-heavy
                #if node is right-heavy: zig-zag
                if node_balance < 0:
                    #print('left-right')
                    self.rotate(node.right.key)
                    self.rotate(node.parent.key)
                else:
                    self.rotate(node.key)
            elif parent_balance < -1: #right-heavy
                #if node is left-heavy: zig-zag
                if node_balance > 0:
                    #print('right-left')
                    self.rotate(node.left.key)
                    #print(self)
                    #print(node, node.parent)
                    self.rotate(node.parent.key)
                    #print(self)
                else:
                    self.rotate(node.key)
            node = node.parent
        self.updateHeight(self.root)
        #print(self)

    def checkAVL(self, node):
        self.updateHeight(node)
        if abs(self.height(node.left) - self.height(node.right)) <= 1:
            print('Root node subtrees have height {} and {}, satisfying AVL.'.format(self.height(node.left), self.height(node.right)))
            return True
        print('Nope! Root node subtrees have height {} and {}.'.format(self.height(node.left), self.height(node.right)))
        return False

    def check(self):
        return super().check() and self.checkAVL(self.root)

class BTree(Tree):
    pass

class FusionTree(BTree):
    '''A B-tree with branching factor w**(1/5). Distinguishes k=O(w**1/5) keys by thinking of them as a path denoted by a bitstring. Look at the branching nodes to distinguish leaves.
    '''
    pass

class VanEmdeBoas():
    pass

class FusionTree():
    pass

class Quadtree(Tree):
    ''''''
    pass

class Octree(Tree):
    ''''''
    pass

class PrefixTree(Tree):
    '''Also known as a Trie.'''
    pass

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

class Test(object):
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

        splay_series = pd.Series(splay_series)
        bst_series = pd.Series(bst_series)

        bst_series.plot(alpha=0.2)
        splay_series.plot(alpha=0.2)

        bst_series.ewm(span=int(n/m)).mean().plot(style='g--')
        splay_series.ewm(span=int(n/m)).mean().plot(style='k--')

        plt.title("BST time: {0}  Splay tree time: {1}".format(int((t1-t0)*1000)/1000,int((t2-t1)*1000)/1000))
        plt.show()

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
        df = pd.DataFrame()
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

#TODO: write a method that searches for the times when a splay tree is better
#than a regular BST
'''

functions = inspect.getmembers(Test, inspect.isfunction)
classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
print('Test class functions: ', ', '.join([f[0] for f in functions]))
print('Classes: ', ', '.join([c[0] for c in classes]))
print('Currently implemented: SplayTree and BinarySearchTree')

avl = AVLTree()
for key in [41, 65, 50, 20, 11, 29, 26, 23]:
    avl.insert(key)
