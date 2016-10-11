from trees import *
import matplotlib.pyplot as plt
import pandas as pd

#size and styling of the output graph
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.style.use('fivethirtyeight')

class Test(object):
    def __init__(self, tree_dict=None):
        if not tree_dict:
            self.tree_dict = {'BST':BinarySearchTree(), 'SplayTree':SplayTree()}
        else:
            self.tree_dict = tree_dict

    def insertRandomElements(self, n=100):
        seq = [i for i in range(n)]
        np.random.shuffle(seq)
        for name,tree in self.tree_dict.items():
            print('inerting {} shuffled values into {}'.format(n, name))
            for num in seq:
                tree.insert(num)
        return seq

    def treeCompare(self, load_seq, access_seq):
        for name,tree in self.tree_dict.items():
            for item in load_seq:
                tree.insert(item)
        for name,tree in self.tree_dict.items():
            t0 = time.time()
            for item in access_seq:
                tree.search(item)
            t1 = time.time()
            print("{0} took {1} seconds. Height: {2}".format(name, int((t1-t0)*1000)/1000, tree.height()))

    def dynamicOptimalityRace(self, n=10000, subset_percent=1):
        seq = self.insertRandomElements(n)

        for name,tree in self.tree_dict.items():
            print('levels of {}'.format(name))
            for lvl in range(tree.height()):
                print('level {}: {}'.format(lvl, len(tree.getLevel(lvl))))
        names = list(self.tree_dict.keys())
        names = ','.join(names[:-1]) + ', and ' + names[-1]
        print('Racing {} on access times for a reapeating subset of {}\% of {}k total inserted elements.'.format(names, subset_percent, int(n/100)/10))

        np.random.shuffle(seq)
        series = {}
        times = {}

        #subset = [num for num in seq if num < m]
        #subset = sorted(seq)[int(n/2):int(n/2)+m]
        m = int(n * (subset_percent/100))
        subset = seq[:m]
        print('subset (length {}): {}'.format(len(subset), subset))
        for name,tree in self.tree_dict.items():
            series[name] = []
            times[name] = []
            for i in range(10):
                np.random.shuffle(subset)
                for num in subset:
                    t0 = time.time()
                    node = tree.search(num)
                    times[name].append(time.time() - t0)
                    node_depth = tree.depth(node)
                    series[name].append(node_depth)

        for name,tree in self.tree_dict.items():
            tree_series = pd.Series(series[name])
            tree_series.plot(alpha=0.1, label='{} depth'.format(name))
            tree_series.ewm(span=int(n/m)).mean().plot(style='--', label='{} depth (smoothed)'.format(name))

            time_series = pd.Series(times[name]) * 100 * 1000
            time_series.plot(label='{} access time'.format(name))

        plt.legend()
        plt.title("{}".format(names))
        plt.show()

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

test = Test({'AVLTree':AVLTree(), 'BST':BinarySearchTree(), 'SplayTree': SplayTree()})
test.dynamicOptimalityRace()
