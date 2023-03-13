from enum import Enum


class State:
    def __init__(self, type='normal', reward=0):
        """Initialize Field object.

        Args:
            type (str, optional): Type of the field, can be: 'start', 'normal', 'special', 'forbidden', 'terminal'. Defaults to 'normal'.
        """
        self._types = ['start', 'normal', 'special', 'forbidden', 'terminal']
        self._actions = ['up', 'down', 'left', 'right']
        self._type = type
        self._reward = reward
        self._value = 0
        self._action = None
        self._statevalues = [0]


    @property
    def reward(self):
        return self._reward
    
    @property
    def type(self):
        return self._type
    
    @property
    def value(self):
        return self._value
    
    @property
    def action(self):
        return self._action
    
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