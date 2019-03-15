import random
from src.utility import ReplayBuffer as rpb


def ran_trans():
    """
    creates random transition batches
    """
    s = random.randint(1,101)
    a = random.randint(1,101)
    r = - random.randint(1,101)
    s_2 = random.randint(1,101)
    return s, a, r, s_2


buffer0 = rpb.ReplayBuffer(5)
transitions = [ran_trans() for i in range(10)]


assert(buffer0.count == 0)

buffer0.add(*transitions[0].get())

assert(buffer0.count ==1)

for trans in transitions:
    buffer0.add(*trans.get())

assert(buffer0.count == 5 and len(transitions) > buffer0.buffer_size)
