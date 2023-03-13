from world import World


class Solver:
    def __init__(self, config='world.txt', verbose=True):
        self.world = World(config=config)
        self.verbose = verbose
        if self.verbose:
            self.world.show_world('type')
            self.world.show_world('reward')
            self.world.show_world('value')

    def solve(self, iter=100, delta=0.01, show=False):
        dv = float('inf')
        i = 0
        while dv >= delta and i < iter:
            dv = self.world.update_values(show=show)
            # print(i)
            i += 1
        print(f"Number of iterations: {i}")

    def show(self, toshow='value'):
        self.world.show_world(toshow, decimal=2)


solver = Solver(config='map2.txt')
solver.solve(iter=100, delta=0.0, show=False)
solver.show()