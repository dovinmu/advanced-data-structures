from trees import BinarySearchTree, SplayTree, AVLTree
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
import time
import timeit
import functools

class TreeTester(object):
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

    def compareDepthAccessTimes(self, n = 1000):
        '''
        Plot the time it takes to access an element by depth
        Returns: pandas dataframe
                 columns: tree
                 rows: time in microseconds
        '''
        k = max([tree.height() for tree in self.tree_dict.values()])
        tree_dict = {}
        for name, tree in self.tree_dict.items():
            samples_max = {}
            samples_mean = {}
            print('{}: timing access from depth 0 to depth {}'.format(name, k))
            for i in range(k):
                tree_series = []
                total_time = 0
                times_sampled = 100
                level = tree.getLevel(i)
                if level:
                    tree_key = random.choice(level).key
                else:
                    break

                runtime = timeit.timeit(functools.partial(tree.search, tree_key), number = 1)
                samples_mean[i] = (runtime)

            # tree_dict[name+'_max'] = samples_max
            tree_dict[name+'_mean'] = samples_mean
        return DataFrame(tree_dict)

'''
#interesting attempt to print tree structure
#check out http://blog.mikedll.com/2011/04/red-black-trees-in-python.html
#for possible awesome console based printing of trees

#TODO: write a method that searches for the times when a splay tree is better
#than a regular BST
'''

# functions = inspect.getmembers(Test, inspect.isfunction)
# classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
# print('Test class functions: ', ', '.join([f[0] for f in functions]))
# print('Classes: ', ', '.join([c[0] for c in classes]))
# print('Currently implemented: SplayTree, BinarySearchTree, AVLTree')
#
# test = Test({'AVLTree':AVLTree(), 'BST':BinarySearchTree(), 'SplayTree': SplayTree()})
# test.compareDepthAccessTimes()
