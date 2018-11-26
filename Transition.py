import numpy as np


class Transition:
    """
    Modeling a simple MDP transition model
    given state, action, reward and state'
    """

    def __init__(self, s, a, r, s_2):
        self.s = s
        self.a = a
        self.r = r
        self.s_2 = s_2

    def __str__(self):
        return  " ".join(str(i) for i in [self.s, self.a, self.r, self.s_2])

    def get(self):
        return (self.s, self.a, self.r, self.s_2)
