import sys

sys.path.append('../')

import random

import numpy as np

import ReplayBuffer as rpb
import Transition as trans

def ran_trans():
    """
    creates random transition tupel
    """
    s = random.randint(1,101)
    a = random.randint(1,101)
    r = - random.randint(1,101)
    s_2 = random.randint(1,101)
    return trans.Transition(s, a, r, s_2)

buffer0 = rpb.ReplayBuffer(5)
transitions = [ran_trans() for i in range(10)]


assert(buffer0.count == 0)

buffer0.add(*transitions[0].get())

assert(buffer0.count ==1)

for trans in transitions:
    buffer0.add(*trans.get())

assert(buffer0.count == 5 and len(transitions) > buffer0.buffer_size)
