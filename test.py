from trees import *
import matplotlib.pyplot as plt
import pandas as pd

#size and styling of the output graph
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.style.use('fivethirtyeight')

class Test(object):



    def treeCompare(self, load_seq, access_seq):
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

    def treeRace(self, n=100000, subset_percent=.1):
        bst = BinarySearchTree()
        splay = SplayTree()

        seq = [i for i in range(n)]
        np.random.shuffle(seq)
        print('inerting {} values'.format(n))
        for num in seq:
            bst.insert(num)
            splay.insert(num)
        #print(bst)
        print('levels of the splay tree')
        for lvl in range(splay.height()):
            print('level {}: {}'.format(lvl, len(splay.getLevel(lvl))))


        print('Racing Splay Tree vs Binary Search Tree (unbalanced, but randomly built) on access times for a reapeating subset of {}\% of {}k total inserted elements.'.format(subset_percent, int(n/100)/10))
        np.random.shuffle(seq)
        splay_series = []
        splay_time = []
        bst_series = []
        bst_time = []
        #subset = [num for num in seq if num < m]
        #subset = sorted(seq)[int(n/2):int(n/2)+m]
        m = int(n * (subset_percent/100))
        subset = seq[:m]
        print('subset (length {}): {}'.format(len(subset), subset))
        for i in range(10):
            np.random.shuffle(subset)
            for num in subset:
                node = bst.search(num)
                bst_depth = bst.depth(node)
                bst_series.append(bst_depth)
                t0 = time.time()
                bst.search(num)
                bst_time.append(time.time() - t0)
        for i in range(10):
            np.random.shuffle(subset)
            for num in subset:
                node = splay.search(num, splay=False)
                splay_depth = splay.depth(node)
                splay_series.append(splay_depth)
                t0 = time.time()
                splay.search(num)
                splay_time.append(time.time() - t0)

        splay_series = pd.Series(splay_series)
        bst_series = pd.Series(bst_series)

        bst_series.plot(alpha=0.2, label='BST depth')
        splay_series.plot(alpha=0.2, label='Splay depth')

        bst_series.ewm(span=int(n/m)).mean().plot(style='g--', label='BST depth (smoothed)')
        splay_series.ewm(span=int(n/m)).mean().plot(style='k--', label='Splay depth (smoothed)')

        splay_time = pd.Series(splay_time) * 100000
        bst_time = pd.Series(bst_time) * 100000
        splay_time.plot(label='Splay access time')
        bst_time.plot(label='BST access time')
        plt.legend()
#        plt.title("BST time: {0}  Splay tree time: {1}".format(int((t1-t0)*1000)/1000,int((t2-t1)*1000)/1000))
        plt.show()

        print('levels of the splay tree')
        elements = 0
        for lvl in range(splay.height()):
            num = len(splay.getLevel(lvl))
            elements += num
            print('level {}: {}'.format(lvl, num))
        print('elements: {}'.format(elements))

    def compareDepthAccessTimes(self, n = 10000):
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
        #print(bst)
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
print('Currently implemented: SplayTree, BinarySearchTree, AVLTree')

test = Test()
#test.compareDepthAccessTimes(1000000)
test.treeRace()
