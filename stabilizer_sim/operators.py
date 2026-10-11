from typing import Union, List
import numpy as np


class Operator:
    def __init__(self, dim: Union[int, List[int]]):

        if isinstance(dim, (int, np.integer)):
            self.dim = dim
            self.rank = 1
        elif isinstance(dim, (list, tuple, np.ndarray)):
            self.dim = dim
            self.rank = len(dim)
        else:
            raise TypeError('dim must be of type int, list, or np.ndarray. received type', type(dim))

        # TODO: Currently no H and no phase
        self.rels = {
            ('X', 'Z'): 'Y',
            ('Z', 'X'): 'Y',
            ('X', 'Y'): 'Z',
            ('Y', 'X'): 'Z',
            ('Y', 'Z'): 'X',
            ('Z', 'Y'): 'X',
        }

        self.ops = np.full(self.dim, 'I', dtype=str)

    def __getitem__(self, idx: int):
        return self.ops[idx]

    def __repr__(self):
        return str(self.ops)

    def __mul__(self, other):

    #     assert isinstance(self.ops, np.ndarray) and isinstance(other.ops, np.ndarray), \
    # (type(self.ops), type(other.ops))
        
        a, b = self.ops, other.ops
        out = np.full(a.shape, 'I', dtype=str)

        a_id, b_id = (a == 'I'), (b == 'I')
        same = (a == b)

        out[a_id & ~b_id] = b[a_id & ~b_id]      # a is I, take b
        out[b_id & ~a_id] = a[b_id & ~a_id]      # b is I, take a

        rest = ~(a_id | b_id | same)             # both non-I and different
        out[rest] = [self.rels[(x, y)] for x, y in zip(a[rest], b[rest])]

        new = Operator(self.dim)
        new.ops = out
        return new

    def copy(self):

        new = Operator(self.dim)
        new.ops = self.ops

        return new


class _Single(Operator):
    label = None

    def __init__(self, dim, positions):
        super().__init__(dim)
        pos = np.asarray(positions, dtype=int)
        if pos.size == 0:
            return 
        if self.rank == 1:
            self.ops[pos.ravel()] = self.label
        else:
            self.ops[tuple(np.atleast_2d(pos).T)] = self.label

class X(_Single): label = 'X'
class Y(_Single): label = 'Y'
class Z(_Single): label = 'Z'
class H(_Single): label = 'H'