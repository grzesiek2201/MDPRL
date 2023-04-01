from world import World
from functools import wraps
from time import time
import matplotlib.pyplot as plt
import numpy as np
import sys


def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        t0 = time()
        result = f(*args, **kwargs)
        t1 = time()
        print(f"func: {f.__name__}; args: [{args, kwargs}]; took: {t1-t0} sec")
        return result
    return wrap


class MDPSolver:
    def __init__(self, config='world.txt', y=None, e=None, verbose=True):
        self.world = World(config=config, y=y, e=e)
        self.verbose = verbose
        self.trials = []
        if self.verbose:
            self.world.show_world('type')
            self.world.show_world('reward')
            self.world.show_world('value')

    @timeit
    def solve(self, iter=100, delta=0.001, show=False):
        dv = float('inf')
        i = 0
        while dv >= delta and i < iter:
            values = [state.value for state in self.world.states.flatten()]
            self.trials.append(values)
            dv = self.world.update_values(show=show)
            i += 1
        print(f"Number of iterations: {i}")

    def plot_trials(self):
        data = [np.vstack(self.trials)[:,i] for i in range(len(self.trials[0]))]
        plt.figure(figsize=(7, 7))
        for state in data:
            plt.plot(state)
        plt.legend([self.world._cast_index(index) for index in np.ndindex(self.world.size)])
        plt.grid()
        plt.show()

    def show(self, toshow='value', decimal=2):
        self.world.show_world(toshow, decimal=decimal)

    def show_policy(self):
        self.world.show_policy()

    def show_graph(self):
        self.world.show_graph()



def main(y=None, e=None, iter=100000):
    solver = MDPSolver(config='world.txt', y=y, e=e, verbose=False)
    solver.solve(iter=iter, delta=0.0001, show=False)
    solver.show(toshow='type')
    solver.show(toshow='reward')
    solver.show(decimal=4)
    solver.show_policy()
    # solver.show_graph()
    # solver.plot_trials()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(float(sys.argv[1]))
    elif len(sys.argv) == 3:
        main(float(sys.argv[1]), float(sys.argv[2]))
    elif len(sys.argv) == 4:
        main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    else:
        print("Wrong number of input arguments, exiting.")