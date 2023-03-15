from enum import Enum


class State:
    def __init__(self, type='normal', reward=0):
        """Initialize Field object.

        Args:
            type (str, optional): Type of the field, can be: 'normal', 'special', 'forbidden', 'terminal'. Defaults to 'normal'.
        """
        self._types = ['normal', 'special', 'forbidden', 'terminal']
        self._actions = ['up', 'down', 'left', 'right']
        self._type = type
        self._reward = reward
        self._value = 0
        self._values = [0, 0, 0, 0]  # up down left right
        self._action = None
        self._statevalues = [0]  # biggest historical values of the state
        self._policies = ['^', 'v', '<', '>']
        self._policy = 'o'
        self._nactions_taken = [0, 0, 0, 0]  # number of actions taken in this state ['up', 'down', 'left', 'right']

    def set_values(self, val):
        """Set all values to one value

        Args:
            val (float): value
        """
        self._values = [val for _ in range(len(self._values))]

    @property
    def reward(self):
        return self._reward
    
    @property
    def type(self):
        return self._type
    
    @property
    def value(self):
        return self._value
        # return np.max(self._values)
    
    @property
    def action(self):
        return self._action
    
    @property
    def statevalues(self):
        return self._statevalues
    
    @property
    def policy(self):
        return self._policy

    @property
    def values(self):
        return self._values

    @reward.setter
    def reward(self, val):
        self._reward = val

    @type.setter
    def type(self, val):
        if val in self._types:
            self._type = val

    @value.setter
    def value(self, val):
        self._statevalues.append(val)
        self._value = val

    @value.setter
    def action(self, val):
        if val in self._actions:
            self._action = val

    @policy.setter
    def policy(self, val):
        if val in self._policies:
            self._policy = val

    @values.setter
    def values(self, vallist):
        self._values = vallist