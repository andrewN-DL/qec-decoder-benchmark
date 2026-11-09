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
        # self.rels = {
        #     ('X', 'Z'): 'Y',
        #     ('Z', 'X'): 'Y',
        #     ('X', 'Y'): 'Z',
        #     ('Y', 'X'): 'Z',
        #     ('Y', 'Z'): 'X',
        #     ('Z', 'Y'): 'X',
        # }

        self.ex = np.zeros(self.dim, dtype=np.uint8)
        self.ez = np.zeros(self.dim, dtype=np.uint8)

    def __getitem__(self, idx: int):
        return self.ex[idx], self.ez[idx]

    def __repr__(self):
        return str('X', self.ex, 'Z', self.ez)

    def __mul__(self, other):

        #     assert isinstance(self.ops, np.ndarray) and isinstance(other.ops, np.ndarray), \
        # (type(self.ops), type(other.ops))

        outx = self.ex ^ other.ex
        outz = self.ex ^ other.ez

        new = Operator(self.dim)
        new.ex, new.ez = outx, outz
        return new

    def copy(self):
        new = Operator(self.dim)
        new.ex, new.ez = self.ex, self.ez

        return new


class _Single(Operator):
    label = None

    def __init__(self, dim, positions):
        super().__init__(dim)
        pos = np.asarray(positions, dtype=int)
        if pos.size == 0:
            return
        if self.rank == 1:
            self.ez[pos.ravel()], self.ex[pos.ravel()] = self.label[0], self.label[1]
        else:
            idx = tuple(np.atleast_2d(pos).T)
            self.ez[idx], self.ex[idx] = self.label[0], self.label[1]


class X(_Single): label = [0, 1]
class Y(_Single): label = [1, 1]
class Z(_Single): label = [1, 0]
# class H(_Single): label = 'H'