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
        '''Print tree structure (needs work)'''
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

class Tree(object):
    def __init__(self):
        self.root = None

class BinaryTree(Tree):
    def depth(self):
        leaves = [self.root]
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
    def search(self, x):
        #accessing a node x brings it to the root
        pass
class RangeTree(object):
    pass

bst = BinarySearchTree()
bst.insert(1)
bst.insert(2)
bst.insert(0)
bst.insert(3)
bst.insert(4)
bst.insert(6)
