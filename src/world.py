from state import State
import numpy as np
from pathlib import Path
import re
import copy
import matplotlib.pyplot as plt
import random


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
    def __init__(self, config='world.txt', y=None, e=None):
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
        if y is not None:
            if not 0 <= y <= 1:
                raise Exception(f"Wrong gamma value: {y}")
        self._size = (0, 0)
        self._states = []  # initialize states
        self._p1 = 0
        self._p2 = 0
        self._p3 = 0
        self._p4 = 0
        self._r = 0
        self._y = y
        self._e = e
        self._actor = np.array([0, 0])  # actor's position on the map in terms of matrix indices
        self._actions = [0, 1, 2, 3] # ['forward', 'left', 'right', 'backward']
        self._directions = [0, 1, 2, 3]  # ['up', 'down', 'left', 'right']

        checksum = self._init_world(filename=config)

        self._reinit_world(checksum)

    def _init_states(self):
        states = np.array([State() for _ in range(self.size[0] * self.size[1])])
        self._states = states.reshape(self.size[0], self.size[1])

    def _reinit_world(self, checksum):

        if not checksum[0]:
            raise Exception("World size not provided.")

        if not checksum[1]:
            raise Exception("Probabilities not provided.")

        if not checksum[2]:
            raise Exception("Rewards not provided.")

        # if start position not initialized
        if not checksum[3]:
            self.start = np.array(random.randint(0, self.size[0]), random.randint(0, self.size[1]))
            self._actor = self.start

        if self._y is None:
            raise Exception("Gamma not set.")

        if self._e is None:
            raise Exception("Epsilon not set.")


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
        checksum = [False] * 4
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
                checksum[0] = True
            elif params[0] == 'S':
                self.start = np.array((self.size[0] - int(params[2]), int(params[1]) - 1))
                self._actor = np.array((self.size[0] - int(params[2]), int(params[1]) - 1))
                checksum[3] = True
            elif params[0] == 'P':
                self._set_probability(float(params[1]), float(params[2]), float(params[3]))
                checksum[1] = True
            elif params[0] == 'R':
                if not checksum[0]:
                    raise Exception("World size not provided.")
                self._r = float(params[1])
                # update reward values
                for state in self._states.flatten():
                    if state.type == 'N':
                        state.reward = self._r
                checksum[2] = True
            elif params[0] == 'G':
                if self._y is None:
                    self.y = float(params[1])
                if not 0 <= self._y <= 1:
                    raise Exception(f"Wrong gamma value: {self._y}") 
            elif params[0] == 'E':
                if self._e is None:
                    self._e = float(params[1])
            elif params[0] == 'T':
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'T'
                self._states[-(int(params[2])), int(params[1]) - 1].reward = float(params[3])
            elif params[0] == 'B':
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'S'
                self._states[-(int(params[2])), int(params[1]) - 1].reward = float(params[3])
            elif params[0] == 'F':
                self._states[-(int(params[2])), int(params[1]) - 1].type = 'F'
                self._states[-(int(params[2])), int(params[1]) - 1]._value = 0
                self._states[-(int(params[2])), int(params[1]) - 1].reward = 0

        return checksum

    def _set_world_size(self, n, m):
        self.size = (n, m)
        self._init_states()

    def _set_probability(self, p1, p2, p3):
        if 0 <= p1 + p2 + p3 <= 1:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self._p4 = round(1 - p1 - p2 - p3, 3)  # rounded so there's no insignificant small value like 1e-15
            return
        raise Exception("Probabilities sum up to more than 1.")

    def _move(self, dir, position=None):
        """Moves the actor in the desired direction according to probability distribution
    
        Args:
            dir (int): 0, 1, 2, 3 ('up', 'left', 'right', 'down')
            action (int): 0, 1, 2, 3 ('forward', 'left', 'right', 'backward')
            position (np.ndarray[int, int]): current position
        Returns:
            list[int, int], float: position, value in new state
        """
        prob = random.random()
        if prob < self.p1:
            action = 0
        elif prob < self.p1 + self.p2:
            action = 1
        elif prob < self.p1 + self.p2 + self.p3:
            action = 2
        else:
            action = 3
        movement = self._next_state(dir, action)
        next_pos = position + movement
        if any((0, 0) > next_pos) or any(self.size <= next_pos):
            new_state = self.get_state(position)
            return position, new_state.value
        else:
            new_state = self.get_state(next_pos)
            if new_state.type == 'F':
                new_state = self.get_state(position)
                return position, new_state.value
            return next_pos, new_state.value

    def _next_state(self, desired, action):
        """Says where to move based on orientation and action

        Args:
            desired (int): desired movement: [0, 1, 2, 3] (up, down, left, right)
            action (int): type of movement: [0, 1, 2, 3] (forward, left, right, backward)

        Returns:
            numpy.ndarray: list of indices to add to current position
        """
        # want to go up
        if desired == 0:
            if action == 0:
                return UP_I  # 'up'
            elif action == 1:
                return LEFT_I  # 'left'
            elif action == 2:
                return RIGHT_I  # 'right'
            else:
                return DOWN_I  # 'down'
        # want to go down
        elif desired == 1:
            if action == 0:
                return DOWN_I # 'down'
            elif action == 1:
                return RIGHT_I # 'right'
            elif action == 2:
                return LEFT_I # 'left'
            else:
                return UP_I # 'up'
        # want to go left
        elif desired == 2:
            if action == 0:
                return LEFT_I # 'left'
            elif action == 1:
                return DOWN_I # 'down'
            elif action == 2:
                return UP_I # 'up'
            else:
                return RIGHT_I # 'right'
        # want to go right
        elif desired == 3:
            if action == 0:
                return RIGHT_I # 'right'
            elif action == 1:
                return UP_I # 'up'
            elif action == 2:
                return DOWN_I # 'down'
            else:
                return LEFT_I # 'left'

    def update_Q(self):
        pos = self._actor  # current position of the actor
        state = self.get_state(pos)  # current state
        if state.type == 'T':
            state._nactions_taken[0] += 1
            state.set_values(state.reward)
            state.value = np.max(state.values)
            self._actor = self.start
            return True
        max_move = np.argmax(state.values)
        prob = random.random()
        if prob < self._e:
            move = random.randint(0, 3)
        else:
            move = max_move
        state._nactions_taken[move] += 1
        dir = self._directions[move]  # get move direction as a string
        new_pos, new_value = self._move(dir=dir, position=pos)  # calculate new position and reward in that position
        self._actor = new_pos  # update actor's position
        rhs = 1/state._nactions_taken[move] * (state.reward + self._y * new_value - state.value)  # value of right-hand-side of the equation
        state.values[move] += rhs
        state.value = np.max(state.values)
        return False

    def alpha(self, N, move):
        return 1/N[move]

    def print_q(self):
        for index in np.ndindex(self.size):
            state = self.get_state(index)
            print(f"{self._cast_index(index)}\t up:{state.values[0]};\t down:{state.values[1]};\t left:{state.values[2]};\t right:{state.values[3]}")

    def update_values(self, show=False):
        """Method used to update values of inidivudal states in the MDP Value Iteration solver.

        Args:
            show (bool, optional): Show iterative values. Defaults to False.

        Returns:
            float: Maximum value error.
        """
        dv = []
        actions = [0 for _ in range(4)]  # array to hold actions
        next_states = [0 for _ in range(4)]  # array to hold next states
        states_values = []
        for index in np.ndindex(self.size):  # take every state
            state = self.get_state(index)  # real state
            if state.type in ('T', 'F'):
                v0 = state.value
                v = state.reward
                dv.append(np.abs(v0 - v))
                states_values.append([v for _ in range(4)])
                continue
            v0 = state.value
            values = []

            for direction in self._directions:  # for each action - up down left right
                actions[0] = self._next_state(direction, 0)  # go forward in selected direction
                actions[1] = self._next_state(direction, 1)
                actions[2] = self._next_state(direction, 2)
                actions[3] = self._next_state(direction, 3)
                # check if actions are legal and calculate next state for each action
                for i, action in enumerate(actions):
                    if (any((0, 0) > index + action) or 
                            any(self.size <= index + action) or 
                            self.get_state(index + action).type == 'F'):
                        next_states[i] = index
                    else:
                        next_states[i] = index + action  # next states in order: forward, left, right, backward (p1, p2, p3, p4)
                cost = self.bellman_rule(index=index, surrounding_indices=next_states)
                values.append(cost)  # calculate value for new state
            v = np.max(values)  # value of the utility in the direction of the best move
            states_values.append(values)
            dv.append(np.abs(v0 - v))
        for i, state in enumerate(self.states.flatten()):
            state.values = states_values[i]
            state.value = np.max(states_values[i])
        if show:
            self.show_world(toshow='value')
        return np.max(dv)

    def bellman_rule(self, index, surrounding_indices):

        probabilities = [self.p1, self.p2, self.p3, self.p4]
        values = [self.get_state(index).value for index in surrounding_indices]
        reward = self.get_state(index).reward
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

    def _cast_policy(self, val, state=None):
        """Casts number (0,1,2,3 == up,down,left,right) to direction symbol.

        Args:
            val (int): number of direction

        Returns:
            str: direction symbol
        """
        if state:
            if state.type in ['N', 'S']:
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
        self._y = val