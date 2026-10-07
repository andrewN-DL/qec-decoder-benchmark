import numpy as np

import operators as o
import decoding.utils as d
import decoding.nine_bit_rep_decoding as nr
import decoding.three_bit_rep_decoding as tr



        

class ThreeQubitRepititionCode(Code):
    def __init__(self, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.intial_state = o.Operator(3)
        
        self.stabs = {    
            'S1': o.Z(3, [0, 1]),
            'S2': o.Z(3, [1, 2]),
        }
    
        self.num_stabilizers = 2


    def run(
            self,
            rounds: int,
            shots: int,
            px: float,
            pm: float,
    ):
        
        # NOTE: PyMatching does not allow a spatial and temporal edge on the same node.
        # This means for pm > px, pz we can get suboptimal matches. Could work this into the graph based on  probs
        self.matching_graph = tr.build_3_qubit_matching_graph(rounds, px, pm)

        result = 0
        for _ in range(shots):
            
            shot_result = self.run_shot(rounds, px, pm)
            result += shot_result

        return result/shots


    def run_shot(
            self,
            rounds: int,
            px: float,
            pm: float,
    ):
        
        syndrome = []
        true_phys_error = np.zeros(3)
        state = self.intial_state.copy()

        for _ in range(rounds):

            # ERRORS 
            errors = self.rng.random(3) < px
            indices = [idx for idx, i in enumerate(errors) if i]
            
            state = state * o.X(3, indices)
            # print(state)
        
            true_phys_error[indices] += 1

            t_measurements = []
            meas_errors = []
            
            for stab in self.stabs.keys():
                t_meas = self.get_syndrome_bit(state, self.stabs[stab])
                t_measurements.append(t_meas)

            m_errors = self.rng.random(2) < pm
            meas_errors.append(m_errors)
            t_measurements = [int(a) ^ int(b) for a, b in zip(m_errors, t_measurements)]

            syndrome.append(t_measurements)

        # Getting predicted error
        detections = d.detect_changes(np.array(syndrome))
        matches = d.match_errors(self.matching_graph, detections)
        predicted_errors = tr.extract_3_qubit_predicted_errors(matches)

        # Calculating true errors
        true_total_error_x = [i % 2 for i in true_phys_error]

        # Comparing
        diff = [int(a) ^ int(b) for a, b in zip(true_total_error_x, predicted_errors)]

        # TODO: not differentiating between logical errors and non-code states. Individual block can be decoded with 'logical error' (IIIXXXIII)
        # TODO: Change this to check EpEt modulo stabilizer group
        if (sum(diff) != 0):
            return 1

        if self.verbose:
            print('Syndrome', syndrome)
            print('Detections', detections)
            print('Matches', matches)
            print('Predicted', predicted_errors)
            print('True', true_phys_error['Z'])
            print(sum(diff))

        return 0
