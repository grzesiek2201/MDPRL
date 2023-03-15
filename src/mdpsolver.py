from world import World
from functools import wraps
from time import time


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
    def __init__(self, config='world.txt', verbose=True):
        self.world = World(config=config)
        self.verbose = verbose
        if self.verbose:
            self.world.show_world('type')
            self.world.show_world('reward')
            self.world.show_world('value')

    @timeit
    def solve(self, iter=100, delta=0.001, show=False):
        dv = float('inf')
        i = 0
        while dv >= delta and i < iter:
            dv = self.world.update_values(show=show)
            # print(i)
            i += 1
        print(f"Number of iterations: {i}")

    def show(self, toshow='value', decimal=2):
        self.world.show_world(toshow, decimal=decimal)

    def show_policy(self):
        self.world.show_policy()

    def show_graph(self):
        self.world.show_graph()

solver = MDPSolver(config='map_lecture.txt')
solver.solve(iter=100, delta=0.0001, show=True)
solver.show(decimal=4)
solver.show_policy()
solver.show_graph()