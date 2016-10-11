import time
import numpy as np
import sys
import inspect
import random
import math

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

class QuadtreeNode(TreeNode):
    '''A quadtree node contains the range over which it spans, stores any points it directly contains in its bucket, and stores all other points indirectly through its four quadrant children.
    '''
    def __init__(self, x1, y1, x2, y2, val=True, NW=None, NE=None, SE=None, SW=None, parent=None, data={}):
        key = (x1,y1,x2,y2)
        super().__init__(key, val, parent=parent, data=data)
        self.NW = NW
        self.NE = NE
        self.SW = SW
        self.SE = SE

        #self.topleft = (x1, y1)
        #self.botright =  (x2, y2)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.bucket = []

    def children(self):
        return [self.NW, self.NE, self.SE, self.SW]

    def addPoint(self, x, y):
        if len(self.bucket) < 1:
            self.bucket.append((x,y))
        else:
            xDivide = (self.x2 - self.x1) / 2
            yDivide = (self.y2 - self.y1) / 2
            if x < xDivide:
                if y < yDivide:
                    if not self.NW:
                        self.setChild(QuadtreeNode(self.x1, self.y1, xDivide, yDivide), 'NW')
                    self.NW.addPoint(x, y)
                else:
                    if not self.SW:
                        self.setChild(QuadtreeNode(self.x1, yDivide, xDivide, self.y2), 'SW')
                    self.SW.addPoint(x, y)
            else:
                if y < yDivide:
                    if not self.NE:
                        self.setChild(QuadtreeNode(xDivide, self.y1, self.x2, yDivide), 'NE')
                    self.NE.addPoint(x, y)
                else:
                    if not self.SE:
                        self.setChild(QuadtreeNode(xDivide, yDivide, self.x2, self.y2), 'SE')
                    self.SE.addPoint(x, y)


    def setChild(self, node, which=''):
        if which == '':
            print('setChild() failed: child must be specified.')
            return
        if which=='NW':
            if self.NW and self.NW.parent is self:
                self.NW.parent = None
            self.NW = node
        elif which=='NE':
            if self.NE and self.NE.parent is self:
                self.NE.parent = None
            self.NE = node
        elif which=='SE':
            if self.SE and self.SE.parent is self:
                self.SE.parent = None
            self.SE = node
        elif which=='SW':
            if self.SW and self.SW.parent is self:
                self.SW.parent = None
            self.SW = node
        if node:
            node.parent = self

    def pointCount(self):
        return len(self.bucket) + (self.NW.pointCount() if self.NW else 0) + (self.NE.pointCount() if self.NE else 0) + (self.SE.pointCount() if self.SE else 0) + (self.SW.pointCount() if self.SW else 0)

    def __repr__(self):
        return "self: {}, NW: {}, NE: {}, SE: {}, SW: {}".format(len(self.bucket),
        self.NW.pointCount() if self.NW else 0, self.NE.pointCount() if self.NE else 0, self.SE.pointCount() if self.SE else 0, self.SW.pointCount() if self.SW else 0)

'''TODO: add these functions to all trees (from https://en.wikipedia.org/wiki/Tree_(data_structure))
Enumerating all the items
Enumerating a section of a tree
Pruning: Removing a whole section of a tree
Grafting: Adding a whole section to a tree
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
            if self.root:
                return self.height(self.root)
            return -1
        if 'height' in node.data:
            return node.data['height']
        #print('computing height')
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
        '''Return all leaves at a specified depth'''
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
        '''Check that all nodes have only one or zero parents, and ensure pointers to and from parents match.'''
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

       Conjectured to be dynamically optimal! However, it's only faster than a
       (non-balanced) BST when operating on a tiny subset of elements.
    '''
    def search(self, key, splay=True):
        node = BinarySearchTree.search(self, key)
        if splay:
            self.splay(node)
        return node

    def splay(self, node):
        while node.parent:
            grandparent = node.parent.parent
            if not grandparent:
                node.rotate()
            elif node.isLeft(node.parent) and node.parent.isLeft(grandparent):
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
                node.rotate()
                node.rotate()
        self.root = node

#challenge: given x, cut a tree into two red-black trees with one
#tree's values < x and one trees's values > x
class RedBlackTree(BinarySearchTree):
    def __init__(self):
        raise NotImplementedError()

class AVLTree(BinarySearchTree):

    def __init__(self):
        raise NotImplementedError()

    #AVL property: height difference between left and right subtrees no greater than 1
    def insert(self, key, val=True):
        node = BinarySearchTree.insert(self, key, val, data={'height':0})
        self.fixAVL(node)

    def fixAVL(self, node):
        while node and node.parent:
            self.updateHeight(node.parent)
            parent_balance = self.height(node.parent.left) - self.height(node.parent.right)
            node_balance = self.height(node.left) - self.height(node.right)
            if parent_balance > 1: #parent is left-heavy
                #if node is right-heavy: zig-zag
                if node_balance < 0:
                    #print('left-right')
                    self.rotate(node.right.key)
                    self.rotate(node.parent.key)
                else:
                    self.rotate(node.key)
            elif parent_balance < -1: #parent is right-heavy
                #if node is left-heavy: zig-zag
                if node_balance > 0:
                    self.rotate(node.left.key)
                    self.rotate(node.parent.key)
                else:
                    self.rotate(node.key)
            node = node.parent
        self.updateHeight(self.root)

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
    '''Generalizes binary search trees and is self-balancing. Optimized for reading and writing large blocks of data.'''
    def __init__(self):
        raise NotImplementedError()

class FusionTree(BTree):
    '''A B-tree with branching factor w**(1/5). Distinguishes k=O(w**1/5) keys by thinking of them as a path denoted by a bitstring. Look at the branching nodes to distinguish leaves.
    '''
    def __init__(self):
        raise NotImplementedError()

class VanEmdeBoasTree():
    '''A tree with an associative array with m-bit integer keys. Is able to contain up to 2^m items, and performs all operations in O(log m) time regardless of the number of elements in the tree.

    U: the size of the universe of integer keys that are allowed. Searches and inserts will take O(log log U) time.
    '''
    def __init__(self, U):
        raise NotImplementedError()

        self.summary = None
        self.min = None
        self.max = None
        self.U = U

    def high(self, el, U):
        '''Get the cluster that the element would appear in.'''
        return int(el / math.sqrt(U))

    def low(self, el, U):
        return el % int(math.sqrt(U))

    def delete(self, el):
        raise Exception()
        if el == self.min:
            pass
        pass

    def predecessor(self, el):
        pass

    def successor(self, el):
        pass

    def insert(self, el):
        idx = high(el, self.U)
        if idx not in self.clusters:
            pass #instantiate cluster
        self.clusters[idx].insert(el)

    def delete(self, el):
        pass


class Quadtree(Tree):
    '''A tree in which each internal node has exactly four children, usually to partition two-dimentional space.'''

    def __init__(self, x1, y1, x2, y2):
        raise NotImplementedError()

        self.root = QuadtreeNode(x1, y1, x2, y2)

    def insert(self, x, y, val=True, data={}):
        '''Recursively choose subdivisions of the given space until we find a subdivision that can hold our value.'''
        self.root.addPoint(x,y)

    def search(self, x, y):
        return self.searchRecursively(x, y, self.root)

    def searchRecursively(self, x, y, node):
        if (x,y) in node.bucket:
            return node
        xDivide = (node.x2 - node.x1) / 2
        yDivide = (node.y2 - node.y1) / 2
        if x < xDivide:
            if y < yDivide:
                if node.NW is None:
                    return None
                return self.searchRecursively(x, y, node.NW)
            else:
                if node.SW is None:
                    return None
                return self.searchRecursively(x, y, node.SW)
        else:
            if y < yDivide:
                if node.NE is None:
                    return None
                return self.searchRecursively(x, y, node.NE)
            else:
                if node.SE is None:
                    return None
                return self.searchRecursively(x, y, node.SE)

    def __repr__(self):
        return repr(self.root)

class Octree(Tree):
    ''''''
    def __init__(self):
        raise NotImplementedError()

class PrefixTree(Tree):
    '''Also known as a Trie.'''
    def __init__(self):
        raise NotImplementedError()

class RangeTree(BinarySearchTree):
    '''
    A range tree on a set of 1-dimensional points is a balanced binary search tree on those points. The points stored in the tree are stored in the leaves of the tree; each internal node stores the largest value contained in its left subtree.

    A range tree on a set of points in d-dimensions is a recursively defined multi-level binary search tree. Each level of the data structure is a binary search tree on one of the d-dimensions. The first level is a binary search tree on the first of the d-coordinates. Each vertex v of this tree contains an associated structure that is a (d−1)-dimensional range tree on the last (d−1)-coordinates of the points stored in the subtree of v.
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
