from state import State
import numpy as np
from pathlib import Path
import re
import copy
import matplotlib.pyplot as plt


UP = np.array([0, 1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
DOWN = np.array([0, -1])
UP_I = np.array([-1, 0])
LEFT_I = np.array([0, -1])
RIGHT_I = np.array([0, 1])
DOWN_I = np.array([1, 0])
UP_SYMBOL = '^'
LEFT_SYMBOL = '<'
RIGHT_SYMBOL = '>'
DOWN_SYMBOL = 'v'

VALUE_DECIMAL_PLACES = 2


class World:
    def __init__(self, config='world.txt'):
        """_summary_

        Args:
            N (int): number of rows
            M (int): number of columns
            p1 (float): forward, 0-1
            p2 (float): left, 0-1
            p3 (float): right, 0-1
            r (float): normal reward
            y (float): discount value
            config (str): path to world config file
        """
        self._size = (0, 0)
        self._states = []  # initialize states
        self._p1 = 0
        self._p2 = 0
        self._p3 = 0
        self._p4 = 0
        self._r = 0
        self._y = 0
        self._e = 0
        self._actor = np.array([1, 1])  # actor's position on the map
        self._actions = ['forward', 'left', 'right', 'backward']
        self._directions = ['up', 'down', 'left', 'right']

        self._init_world(filename=config)

    def _init_states(self):
        states = np.array([State() for _ in range(self.size[0] * self.size[1])])
        self._states = states.reshape(self.size[0], self.size[1])

    def _init_world(self, filename):
        """Read file and initiate the world with given config.

        W (obowiązkowy) określa rozmiar świata: poziomy i pionowy (2xINT),
        S (opcjonalny) określa współrzędne stanu startowego (2xINT),
        P (obowiązkowy) określa rozkład prawdopodobieństwa p1 p2 p3 (3xFLOAT),
        R (obowiązkowy) określa domyślną wartość nagrody r (1xFLOAT),
        G (opcjonalny) określa współczynnik dyskontowania γ (1xFLOAT),
        E (opcjonalny) określa współczynnik eksploracji ε (1xFLOAT),
        T (wielokrotny - musi wystąpić 1 lub więcej razy) definiuje pojedynczy stan terminalny: dwie współrzędne i indywidualną wartość nagrody (2xINT+1xFLOAT),
        B (wielokrotny - może wystąpić 0 lub więcej razy) definiuje pojedynczy stan specjalny: dwie współrzędne i indywidualną wartość nagrody (2xINT+1xFLOAT),
        F (wielokrotny - może wystąpić 0 lub więcej razy) definiuje pojedynczy stan zabroniony: dwie współrzędne (2xINT).

        Args:
            filename (str): config file name in the /config directory
        """
        try:
            path = list(Path(__file__).parent.parent.glob(f"config/{filename}"))[0]
            with open(path, 'r') as file:
                data = file.readlines()
        except FileNotFoundError as e:
            print(e)
        except IndexError as e:
            print(e)
        for line in data:
            params = re.split(r'[\n\s]', line)
            if params[0] == 'W':
                self._set_world_size(int(params[2]), int(params[1]))
            elif params[0] == 'S':
                self.start = np.array((int(params[1]) - 1, int(params[2]) - 1))
                #self._actor = np.array((int(params[1]) - 1, int(params[2]) - 1))
            elif params[0] == 'P':
                self._set_probability(float(params[1]), float(params[2]), float(params[3]))
            elif params[0] == 'R':
                self._r = float(params[1])
                # update reward values
                for state in self._states.flatten():
                    if state.type == 'normal':
                        state.reward = self._r
            elif params[0] == 'G':
                self.y = float(params[1])
            elif params[0] == 'E':
                self._e = float(params[1])
            elif params[0] == 'T':
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'terminal'
                self._states[-(int(params[2])), int(params[1]) - 1].reward = float(params[3])
            elif params[0] == 'B':
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'special'
                self._states[-(int(params[2])), int(params[1]) - 1].reward = float(params[3])
            elif params[0] == 'F':
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'forbidden'
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].reward = 0

    def _set_world_size(self, n, m):
        self.size = (n, m)
        self._init_states()

    def _set_probability(self, p1, p2, p3):
        if 0 <= p1 + p2 + p3 <= 1:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self._p4 = round(1 - p1 - p2 - p3, 3)  # rounded so there's no insignificant small value like 1e-15

    def move(self, dir, position=None, action=None, mode='indices'):
        """Moves the actor in the desired direction according to probability distribution
    
        Args:
            dir (str): 'up', 'left', 'right', 'down'
            action (str): 'forward', 'left', 'right', 'backward'
            position (np.ndarray[int, int]): current position
        Returns:
            list[[int, int], float]: new state and reward in that new state
        """
        # (x,y) in the world frame is (-y, x-1) in the matrix indexing
        if mode == 'world':
            action = np.random.choice(self._actions, 1, p=[self.p1, self.p2, self.p3, self.p4])  # draw an action
            movement = self._next_state(dir, action[0])
            if any((1,1) > self._actor + movement) or ((self._actor + movement)[1] > self.size[0]) or ((self._actor + movement)[0] > self.size[1]):  # if new position is outside the map boundaries, don't update position
                pass
            else:
                self._actor += movement
            # self._update_cost  # update the current cost of policy as well as individual states based on new (self._actor) position
        elif mode == 'indices':
            movement = self._next_state(dir, action)
            if any((0, 0) > position + movement) or any(self.size <= position + movement):
                return position
            else:
                return position + movement

    def _next_state(self, desired, action, mode='indices'):
        """Says where to move based on orientation and action

        Args:
            desired (str): desired movement: up, down, left, right
            action (str): type of movement: forward, backward, left, right
            mode (str, optional): world or indices. Defaults to 'indices'.

        Returns:
            _type_: _description_
        """
        if mode == 'world':
            # want to go up
            if desired == 'up':
                if action == 'forward':
                    return UP  # 'up'
                elif action == 'left':
                    return LEFT  # 'left'
                elif action == 'right':
                    return RIGHT  # 'right'
                else:
                    return DOWN  # 'down'
            # want to go down
            elif desired == 'down':
                if action == 'forward':
                    return DOWN # 'down'
                elif action == 'left':
                    return RIGHT # 'right'
                elif action == 'right':
                    return LEFT # 'left'
                else:
                    return UP # 'up'
            # want to go left
            elif desired == 'left':
                if action == 'forward':
                    return LEFT # 'left'
                elif action == 'left':
                    return DOWN # 'down'
                elif action == 'right':
                    return UP # 'up'
                else:
                    return RIGHT # 'right'
            # want to go right
            elif desired == 'right':
                if action == 'forward':
                    return RIGHT # 'right'
                elif action == 'left':
                    return UP # 'up'
                elif action == 'right':
                    return DOWN # 'down'
                else:
                    return LEFT # 'left'
        
        elif mode == 'indices':
            # want to go up
            if desired == 'up':
                if action == 'forward':
                    return UP_I  # 'up'
                elif action == 'left':
                    return LEFT_I  # 'left'
                elif action == 'right':
                    return RIGHT_I  # 'right'
                else:
                    return DOWN_I  # 'down'
            # want to go down
            elif desired == 'down':
                if action == 'forward':
                    return DOWN_I # 'down'
                elif action == 'left':
                    return RIGHT_I # 'right'
                elif action == 'right':
                    return LEFT_I # 'left'
                else:
                    return UP_I # 'up'
            # want to go left
            elif desired == 'left':
                if action == 'forward':
                    return LEFT_I # 'left'
                elif action == 'left':
                    return DOWN_I # 'down'
                elif action == 'right':
                    return UP_I # 'up'
                else:
                    return RIGHT_I # 'right'
            # want to go right
            elif desired == 'right':
                if action == 'forward':
                    return RIGHT_I # 'right'
                elif action == 'left':
                    return UP_I # 'up'
                elif action == 'right':
                    return DOWN_I # 'down'
                else:
                    return LEFT_I # 'left'

    def update_values(self, show=False):
        """Method used to update values of inidivudal states in the MDP Value Iteration solver.

        Args:
            show (bool, optional): Show iterative values. Defaults to False.

        Returns:
            float: Maximum value error.
        """
        # states_copy = copy.deepcopy(self.states)  # deep copy of the states, instead create a list with new values and update at the end of the loop
        dv = []
        new_v = []
        actions = [0 for _ in range(4)]  # array to hold actions
        next_states = [0 for _ in range(4)]  # array to hold next states
        states_values = []
        for index in np.ndindex(self.size):  # take every state
            state = self.get_state(index)  # real state
            if state.type in ('terminal', 'forbidden', 'special'):
                v0 = state.value
                v = state.reward
                # new_v.append(state.reward, 'o'))
                dv.append(np.abs(v0 - v))
                states_values.append([v for _ in range(4)])
                continue
            v0 = state.value
            values = []

            for direction in self._directions:  # for each action - up down left right
                actions[0] = self._next_state(direction, 'forward', mode='indices')  # go forward in selected direction
                actions[1] = self._next_state(direction, 'left', mode='indices')
                actions[2] = self._next_state(direction, 'right', mode='indices')
                actions[3] = self._next_state(direction, 'backward', mode='indices')
                # check if actions are legal and calculate next state for each action
                for i, action in enumerate(actions):
                    if (any((0, 0) > index + action) or 
                            any(self.size <= index + action) or 
                            self.get_state(index + action).type == 'forbidden'):
                        next_states[i] = index
                    else:
                        next_states[i] = index + action  # next states in order: forward, left, right, backward (p1, p2, p3, p4)
                cost = self.bellman_rule(index=index, surrounding_indices=next_states)
                values.append(cost)  # calculate value for new state
            # take action with maximum value and update the state's value
            # id = np.argmax(values)  # direction of the best move
            v = np.max(values)  # value of the utility in the direction of the best move
            # new_v.append((v, self._cast_policy(id)))
            states_values.append(values)
            dv.append(np.abs(v0 - v))
        for i, state in enumerate(self.states.flatten()):
            # state.value = new_v[i][0]
            state.values = states_values[i]
            state.value = np.max(states_values[i])
            # state.policy = new_v[i][1]
        if show:
            self.show_world(toshow='value')
        return np.max(dv)

    def bellman_rule(self, index, surrounding_indices):

        probabilities = [self.p1, self.p2, self.p3, self.p4]
        # rewards = [self.get_state(index).reward for index in surrounding_indices]
        values = [self.get_state(index).value for index in surrounding_indices]
        reward = self.get_state(index).reward
        # new_v = np.sum([probability * ( reward + self._y * value ) for probability, reward, value in zip(probabilities, rewards, values)])
        new_v = np.sum([probability * value for probability, value in zip(probabilities, values)])
        new_v *= self._y
        new_v += reward
        return new_v

    def get_state(self, indices, state=None):
        if state is None:
            state = self.states
        return state[indices[0], indices[1]]

    def show_world(self, toshow='type', decimal=2):
        """Prints world in string format

        Args:
            toshow (str, optional): 'type', 'value', 'reward'. Defaults to 'type'.
            decimal (int, optional): numer of printed values decimal places
        """
        if toshow == 'type':
            world = np.array([state.type for state in self._states.flatten()]).reshape(self.size[0], self.size[1])
        if toshow == 'reward':
            world = np.array([state.reward for state in self._states.flatten()]).reshape(self.size[0], self.size[1])
        if toshow == 'value':
            world = np.array([round(state.value, decimal) for state in self._states.flatten()]).reshape(self.size[0], self.size[1])
        print(world)

    def show_policy(self):
        # policy = np.array([state.policy for state in self.states.flatten()]).reshape(self.size[0], self.size[1])
        policy = np.array([self._cast_policy(np.argmax(state.values), state=state) for state in self.states.flatten()]).reshape(self.size[0], self.size[1])
        print(policy)
        
    def show_graph(self):
        plt.figure(figsize=(6, 6))
        for state in self.states.flatten():
            plt.plot(state.statevalues)
        plt.legend([self._cast_index(index) for index in np.ndindex(self.size)])
        plt.grid()
        plt.show()

    def _cast_policy(self, val, state=None):
        """Casts number (0,1,2,3 == up,down,left,right) to direction symbol.

        Args:
            val (int): number of direction

        Returns:
            str: direction symbol
        """
        if state:
            if state.type == 'normal':
                return self._cast_policy(val)
            else:
                return 'o'
        if val == 0:
            return UP_SYMBOL
        elif val == 1:
            return DOWN_SYMBOL
        elif val == 2:
            return LEFT_SYMBOL
        elif val == 3:
            return RIGHT_SYMBOL
        else:
            return 'o'

    def _cast_index(self, index):
        return index[1] + 1, self.size[0] - index[0]

    def _take_action(self, pos, action):
        if any((0, 0) > pos + action) or any(self.size <= pos + action):
            return pos
        else:
            return pos + action

    @property
    def p1(self):
        return self._p1
    
    @property
    def p2(self):
        return self._p2
    
    @property
    def p3(self):
        return self._p3
    
    @property
    def p4(self):
        return self._p4
    
    @property
    def size(self):
        return self._size

    @property
    def y(self):
        return self._y

    @property
    def states(self):
        return self._states
    
    @property
    def actions(self):
        return self._actions

    @p1.setter
    def p1(self, val):
        if 0 <= val <= 1 and val + self.p2 + self.p3 <= 1:
            self._p1 = val
        else:
            raise Exception('Incorrect probability value.')
        
    @p2.setter
    def p2(self, val):
        if 0 <= val <= 3 and val + self.p1 + self.p3 <= 1:
            self._p2 = val
        else:
            raise Exception('Incorrect probability value.')
    
    @p3.setter
    def p3(self, val):
        if 0 <= val <= 1 and val + self.p1 + self.p2 <= 1:
            self._p3 = val
        else:
            raise Exception('Incorrect probability value.')
        
    @size.setter
    def size(self, size):
        if size[0] > 0 and size[1] > 0:
            self._size = (size[0], size[1])

    @y.setter
    def y(self, val):
        if 0 <= val <= 1:
            self._y = val