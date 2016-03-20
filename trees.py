import time
import numpy as np
import random

class BinaryTreeNode(object):
    def __init__(self, key, val=True, left=None, right=None, parent=None):
        self.key = key
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

    def setLeft(self, node):
        if self.left and self.left.parent == self:
            self.left.parent = None
        self.left = node
        if node:
            node.parent = self

    def setRight(self, node):
        if self.right and self.right.parent == self:
            self.right.parent = None
        self.right = node
        if node:
            node.parent = self

    def isLeft(self, node):
        return node and node.left == self

    def isRight(self, node):
        return node and node.right == self

    def rotate(self):
        if self.parent is None:
            print("cannot rotate on a root!")
            return
        if self == self.parent.left:
            #right rotation
            A = self.left
            B = self.right
            C = self.parent.right
            other = self.parent
            new_parent = self.parent.parent
            if new_parent:
                if new_parent.left == other:
                    new_parent.setLeft(self)
                else:
                    new_parent.setRight(self)
            self.setLeft(A)
            other.setLeft(B)
            other.setRight(C)
            self.setRight(other)
        elif self == self.parent.right:
            #left rotation
            A = self.parent.left
            B = self.left
            C = self.right
            other = self.parent
            new_parent = self.parent.parent
            if new_parent:
                if new_parent.left == other:
                    new_parent.setLeft(self)
                else:
                    new_parent.setRight(self)
            else:
                self.parent = None
            other.setLeft(A)
            other.setRight(B)
            self.setRight(C)
            self.setLeft(other)
        else:
            print("Cannot find self in parent!")


    def __repr__(self, level=0):
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

class BinaryTree(Tree):
    def height(self, node=None):
        if not node:
            node = self.root
        leaves = [node]
        level = 0
        while len(leaves) > 0:
            new_leaves = []
            for leaf in leaves:
                if leaf:
                    new_leaves.append(leaf.left)
                    new_leaves.append(leaf.right)
            level += 1
            leaves = new_leaves
        return level

    def checkPointers(self):
        seen = {}
        leaves = [self.root]
        while leaves:
            new_leaves = []
            for leaf in leaves:
                if leaf.left:
                    if leaf.right in seen:
                        print('Multiple parents: {0}->{1}'.format(leaf.key, leaf.left.key))
                    else:
                        new_leaves.append(leaf.left)
                    if leaf.left.parent != leaf:
                        print('erroneous back-pointer:{0}->{1}'.format(leaf.key, leaf.left.key))
                if leaf.right:
                    if leaf.right in seen:
                        print('Multiple parents: {0}->{1}'.format(leaf.key, leaf.left.key))
                    else:
                        new_leaves.append(leaf.right)
                    if leaf.right.parent != leaf:
                        print('erroneous back-pointer:{0}->{1}'.format(leaf.key, leaf.right.key))
                seen[leaf] = True
            leaves = new_leaves

class BinarySearchTree(BinaryTree):
    def insert(self, key, val=True):
        if self.root is None:
            self.root = BinaryTreeNode(key, val)
        else:
            self.insertRecursively(key, val, self.root)

    def insertRecursively(self, key, val, node):
        if key > node.key:
            if node.right is None:
                node.right = BinaryTreeNode(key, val, parent=node)
            else:
                self.insertRecursively(key, val, node.right)
        else:
            if node.key == key:
                print("Attempting to insert a duplicate... disallowed")
                return
            if node.left is None:
                node.left = BinaryTreeNode(key, val, parent=node)
            else:
                self.insertRecursively(key, val, node.left)

    def search(self, key):
        if self.root is None:
            return None
        return self.searchRecursively(key, self.root)

    def searchRecursively(self, key, node):
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
        ret = "  "*level+repr(node.key)+"\n"
        if node.left:
            ret += self.stringify(node.left, level+1)
        if node.right:
            ret += self.stringify(node.right, level+1)
        return ret

#challenge: given x, cut a tree into two red-black trees with one
#tree's values < x and one trees's values > x
class RedBlackTree(BinarySearchTree):
    '''
    '''
    pass

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
            #print('depth:', self.height(node))
            if node.parent.parent is None:
                #zig step
                node.rotate()
                #print('zig')
            elif (node.isLeft(node.parent) and node.parent.isLeft(node.parent.parent)) or (node.isRight(node.parent) and node.parent.isRight(node.parent.parent)):
                #zig-zig step
                node.parent.rotate()
                node.rotate()
                #print('zig-zig')
            else:
                node.rotate()
                node.rotate()
                #print('zig-zag')
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
    pass


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
    n = 10000
    std_random = np.random.randn(n)
    load = np.copy(std_random)
    print("{} std dev random nums, sorted access".format(n))
    treeCompare(load, sorted(std_random))
    print("{} std dev random nums, shuffled access".format(n))
    np.random.shuffle(std_random)
    treeCompare(load, std_random)

    print('\n')

    uniform_rand = [random.random() for i in range(n)]
    load = np.copy(uniform_rand)
    print("{} uniform random nums, sorted access".format(n))
    treeCompare(load, sorted(uniform_rand))
    print("{} uniform random nums, shuffled access".format(n))
    np.random.shuffle(uniform_rand)
    treeCompare(load, uniform_rand)
    #print("1000 std deviation random numbers, random load / subset access")
    #treeCompare(std_random, std_random[:100]*10)

#bst = BinarySearchTree()
bst = SplayTree()
bst.insert(3)
bst.insert(10)
bst.insert(4)
bst.insert(2)
bst.insert(1)
bst.insert(16)
bst.insert(6)
bst.insert(2.5)
bst.insert(0.5)
bst.insert(4.5)
bst.insert(0)
print(bst)
