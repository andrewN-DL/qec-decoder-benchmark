from typing import Union, List
import numpy as np


class Operator:
    def __init__(self, dim: int):
        self.dim = dim

        # TODO: Currently no H and no phase
        self.rels = {
            ('X', 'Z'): '-Y',
            ('Z', 'X'): 'Y',
            ('X', 'Y'): 'Z',
            ('Y', 'X'): '-Z',
            ('Y', 'Z'): 'X',
            ('Z', 'Y'): '-X',
        }

        self.ops = dim * ['I']

    def __getitem__(self, idx: int):
        return self.ops[idx]

    def __repr__(self):
        return str(self.ops)

    def __mul__(self, other):
        new_ops = []
        for comp in zip(self.ops, other.ops):
            if comp[0] == comp[1]:
                new_ops.append('I')
            elif comp[0] == 'I':
                new_ops.append(comp[1])
            elif comp[1] == 'I':
                new_ops.append(comp[0])
            else:
                new_ops.append(self.rels[comp])

        new = Operator(self.dim)
        new.ops = new_ops

        return new

    def copy(self):

        new = Operator(self.dim)
        new.ops = self.ops

        return new


class X(Operator):
    def __init__(self, dim: int, positions: Union[int, List[int]]):
        super().__init__(dim)

        if isinstance(positions, int):
            positions = [positions]

        for pos in positions:
            self.ops[pos] = 'X'

class Y(Operator):
    def __init__(self, dim: int, positions: Union[int, List[int]]):
        super().__init__(dim)

        if isinstance(positions, int):
            positions = [positions]

        for pos in positions:
            self.ops[pos] = 'Y'

class Z(Operator):
    def __init__(self, dim: int, positions: Union[int, List[int]]):
        super().__init__(dim)

        if isinstance(positions, int):
            positions = [positions]

        for pos in positions:
            self.ops[pos] = 'Z'

class H(Operator):
    def __init__(self, dim: int, positions: Union[int, List[int]]):
        super().__init__(dim)

        if isinstance(positions, int):
            positions = [positions]

        for pos in positions:
            self.ops[pos] = 'H'