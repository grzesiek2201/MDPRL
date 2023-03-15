from world import World
import numpy as np
import matplotlib.pyplot as plt


class QSolver:
    def __init__(self, config='world.txt'):
        self.world = World(config=config)
        self.trials = []

    def solve(self, iter=100, delta=0.1, horizon=100):
        i = 0

        while i < iter:
            if self.world.update_Q():
                values = [state.value for state in self.world.states.flatten()]
                self.trials.append(values)
                i += 1
                if i > horizon and np.sqrt(np.mean(np.power(np.array(self.trials[-horizon - 1]) - np.array(self.trials[-1]), 2))) < delta:
                    break
                if i % 1000 == 0:
                    print(f"iteration {i}")

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

solver = QSolver(config='q_test.txt') 
solver.world.show_world()
solver.solve(iter=100000, delta=0.001, horizon=1000)
solver.world.print_q()
solver.world.show_world(toshow='value')
solver.world.show_policy()
# solver.world.show_graph()
solver.plot_trials()