import numpy as np

import operators as o
from base_code import Code
import decoding.utils as d
import decoding.nine_bit_rep_decoding as nr
import decoding.three_bit_rep_decoding as tr


class ToricCode(Code):
    def __inti__(self, dist, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.dist = dist


    def meas_stab_z(self, state, i, j):
        # Face operator (Z1 Z2 Z3 Z4)
        relevant_data_bits = [
            state[2*i, j],
            state[2*i+1, j],
            state[2*i+1, (j+1) % self.dist],
            state[(2*i+2) % (2*self.dist), j],
            ]

        parity = sum([1 for el in relevant_data_bits if el not in ['I', 'X']]) % 2

        return parity
    

    def meas_stab_x(self, state, i, j):
        # Vertex operator (X1 X2 X3 X4)
        relevant_data_bits = [
            state[(2*i-1) % (2*self.dist), j],
            state[2*i, (j-1) % self.dist],
            state[2*i, j],
            state[2*i+1, j],
            ]
        
        parity = sum([1 for el in relevant_data_bits if el not in ['I', 'Z']]) % 2
        
        return parity