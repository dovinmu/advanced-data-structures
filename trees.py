class BinaryTreeNode(object):
    def __init__(self, key, val=True, left=None, right=None):
        self.key = key
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self, level=0):
        ret = "  "*level+repr(self.key)+"\n"
        if self.left:
            ret += self.left.__repr__(level+1)
        if self.right:
            ret += self.right.__repr__(level+1)
        return ret

class BinarySearchTree(object):
    def __init__(self):
        self.root = None

    def insert(self, key, val=True):
        if self.root is None:
            self.root = BinaryTreeNode(key, val)
        else:
            self.insertRecursively(key, val, self.root)

    def insertRecursively(self, key, val, node):
        if key > node.key:
            if node.right is None:
                node.right = BinaryTreeNode(key, val)
            else:
                self.insertRecursively(key, val, node.right)
        else:
            if node.left is None:
                node.left = BinaryTreeNode(key, val)
            else:
                self.insertRecursively(key, val, node.left)

    def search(self, key):
        if self.root is None:
            return None
        return self.searchRecursively(key, self.root)

    def searchRecursively(self, key, node):
        if key == node.key:
            return node.val
        if key > node.key and node.right:
            return self.searchRecursively(key, node.right)
        elif key < node.key and node.left:
            return self.searchRecursively(key, node.left)
        return None

    def __repr__(self):
        if self.root:
            return self.root.__repr__()


class SplayTree(BinarySearchTree):
    def search(self, x):
        #accessing a node x brings it to the root
        pass
class RangeTree(object):
    pass

'''
bst = BinarySearchTree()
bst.insert(1)
bst.insert(2)
bst.insert(0)
bst.insert(3)
bst.insert(4)
bst.insert(6)
bst.insert(-1)
'''
