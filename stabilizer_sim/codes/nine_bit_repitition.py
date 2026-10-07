import numpy as np

import operators as o
from base_code import Code
import decoding.utils as d
import decoding.nine_bit_rep_decoding as nr
import decoding.three_bit_rep_decoding as tr


class NineQubitRepititionCode(Code):
    def __init__(self, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.intial_state = o.Operator(9)
        
        self.stabs = {
            'S1': o.Z(dim=9, positions=[0, 1]),
            'S2': o.Z(9, [1, 2]),
            'S3': o.Z(9, [3, 4]),
            'S4': o.Z(9, [4, 5]),
            'S5': o.Z(9, [6, 7]),
            'S6': o.Z(9, [7, 8]),
            'S7': o.X(9, [0, 1, 2, 3, 4, 5]),
            'S8': o.X(9, [3, 4, 5, 6, 7, 8]),
            }
    
        self.num_stabilizers = 8


    def run(
            self,
            rounds: int,
            shots: int,
            px: float,
            pz: float,
            pm: float,
    ):
        
        # NOTE: PyMatching does not allow a spatial and temporal edge on the same node.
        # This means for pm > px, pz we can get suboptimal matches. Could work this into the graph based on  probs
        self.matching_graph = nr.build_9_qubit_matching_graph(rounds, px, pz, pm)

        result = 0
        for _ in range(shots):
            
            shot_result = self.run_shot(rounds, px, pz, pm)
            result += shot_result

        return result/shots


    def run_shot(
            self,
            rounds: int,
            px: float,
            pz: float,
            pm: float,
    ):
        
        syndrome = []
        true_phys_error = {
            'X': np.zeros(9),
            'Z': np.zeros(9)
        }
        
        state = self.intial_state.copy()

        for _ in range(rounds):

            # ERRORS 
            x_errors = self.rng.random(9) < px
            z_errors = self.rng.random(9) < pz
            x_indices, z_indices = [idx for idx, i in enumerate(x_errors) if i], [idx for idx, i in enumerate(z_errors) if i]
            
            state = state * o.X(9, x_indices)
            # print(state, o.Z(9, z_indices))
            state = state * o.Z(9, z_indices)
            # print(state)
            
            true_phys_error['X'][x_indices] += 1
            true_phys_error['Z'][z_indices] += 1

            t_measurements = []
            meas_errors = []
            
            for stab in self.stabs.keys():
                t_meas = self.get_syndrome_bit(state, self.stabs[stab])
                t_measurements.append(t_meas)

            m_errors = self.rng.random(8) < pm
            meas_errors.append(m_errors)
            t_measurements = [int(a) ^ int(b) for a, b in zip(m_errors, t_measurements)]

            syndrome.append(t_measurements)

        # Getting predicted error
        detections = d.detect_changes(np.array(syndrome))
        matches = d.match_errors(self.matching_graph, detections)
        predicted_x_errors, predicted_z_errors = nr.extract_9_qubit_predicted_errors(matches)

        # Calculating true errors
        true_total_error_x = [i % 2 for i in true_phys_error['X']]

        true_total_error_z = [sum(true_phys_error['Z'][i:i+3]) for i in range(0, len( true_phys_error['Z']), 3)]
        true_total_error_z = [i % 2 for i in true_total_error_z]

        # Comparing
        diff_x = [int(a) ^ int(b) for a, b in zip(true_total_error_x, predicted_x_errors)]
        diff_z = [int(a) ^ int(b) for a, b in zip(true_total_error_z, predicted_z_errors)]

        # TODO: not differentiating between logical errors and non-code states. Individual block can be decoded with 'logical error' (IIIXXXIII)
        # TODO: Change this to check EpEt modulo stabilizer group
        if (sum(diff_x) != 0) | (sum(diff_z) != 0):
            return 1

        if self.verbose:
            print('Syndrome', syndrome)
            print('Detections', detections)
            print('Matches', matches)
            print('Predicted', predicted_x_errors, predicted_z_errors)
            print('True', true_phys_error['Z'])
            print(sum(diff_x), sum(diff_z))

        return 0