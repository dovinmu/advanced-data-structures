import time

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

class SplayTree(BinarySearchTree):
    '''In a splay tree, accessing a node x of a BST brings it to the root.
       Thus, recently accessed items are quick to access again.
       Insertion, look-up and removal are O(logn) amortized. For sequences
       of non-random operations, splay trees perform better than normal BSTs.
    '''
    def search(self, key):
        node = BinarySearchTree.search(self, key)
        self.splay(node)
        return node

    def splay(self, node):
        while node.parent:
            print('depth:', self.height(node))
            if node.parent.parent is None:
                #zig step
                node.rotate()
                print('zig')
            elif (node.isLeft(node.parent) and node.parent.isLeft(node.parent.parent)) or (node.isRight(node.parent) and node.parent.isRight(node.parent.parent)):
                #zig-zig step
                node.parent.rotate()
                node.rotate()
                print('zig-zig')
            else:
                node.rotate()
                node.rotate()
                print('zig-zag')
        self.root = node

class RangeTree(object):
    pass

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
