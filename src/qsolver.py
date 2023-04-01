from world import World
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from time import time
import csv
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


class QSolver:
    def __init__(self, config='world.txt', y=None, e=None):
        self.world = World(config=config, y=y, e=e)
        self.trials = []

    # @timeit
    def solve(self, iter=100000, delta=0.1, horizon=100):
        iter = int(iter)
        i = 0

        values = [state.value for state in self.world.states.flatten()]
        self.trials.append(values)
        
        while i < iter:
            if self.world.update_Q():
                i += 1
                # if i > horizon and np.sqrt(np.mean(np.power(np.array(self.trials[-horizon - 1]) - np.array(self.trials[-1]), 2))) < delta:
                    # break
                if i % 1000 == 0:
                    values = [state.value for state in self.world.states.flatten()]
                    self.trials.append(values)
                    print(f"iteration {i}/{iter}", end='\r')

        print(f"Number of itertions: {i}")
            # i += 1
        # self.world.show_world(toshow='value')

    def plot_trials(self):
        data = [np.vstack(self.trials)[:,i] for i in range(len(self.trials[0]))]
        plt.figure(figsize=(7, 7))
        for state in data:
            plt.plot(state)
        plt.legend([self.world._cast_index(index) for index in np.ndindex(self.world.size)])
        plt.grid()
        plt.show()
        with open('./data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([self.world._cast_index(index) for index in np.ndindex(self.world.size)])
            writer.writerows(self.trials)


def main(y=None, e=None, iter=100000):
    solver = QSolver(config='world.txt', y=y, e=e) 
    solver.world.show_world()
    solver.world.show_world(toshow='reward')
    solver.solve(iter=iter, delta=0, horizon=1000)
    solver.world.print_q()
    solver.world.show_world(toshow='value')
    solver.world.show_policy()
    # solver.world.show_graph()
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