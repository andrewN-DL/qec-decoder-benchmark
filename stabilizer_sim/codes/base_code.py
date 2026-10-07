import numpy as np

class Code:
    def __init__(self, verbose: bool=False, seed: int=None):

        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def get_syndrome_bit(self, state, stabilizer):
        bit = 0
        for comp in zip(state, stabilizer):
            if 'I' not in comp:
                if comp[0] != comp[1]:
                    bit += 1

        return bit % 2