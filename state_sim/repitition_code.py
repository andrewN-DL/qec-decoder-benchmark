from circuit import Circuit
import numpy as np
import copy
from pymatching import Matching

# currently only works for 3-qubit code
class RepititionCode(Circuit):
    def __init__(self, distance: int, seed: int=None, verbose: bool=False):

        super().__init__(distance + 2, seed)
        self.distance = distance
        self.verbose = verbose

        self.initial_state = copy.deepcopy(self.state)

        # TODO: This works for 3 qubit code. For say 9 qubit this would prob need to be programmatically. Shouldn't be too hard
        self.decoding_map = {
            '10': 0,
            '11': 1,
            '01': 2
        }

        self.s1, self.s2 = self.add_stabilizers()

        self.extra_gates = []

    def H(self, bit, extra: bool=True):
        m = super().H(bit, add_to_circuit=False)

        if extra:
            self.extra_gates.append(m)
        return m

    def X(self, bit, extra: bool=True):
        m = super().X(bit, add_to_circuit=False)

        if extra:
            self.extra_gates.append(m)
        return m

    def Z(self, bit, extra: bool=True):
            m = super().Z(bit, add_to_circuit=False)
    
            if extra:
                self.extra_gates.append(m)
            return m

    def S(self, bit, extra: bool=True):
            m = super().S(bit, add_to_circuit=False)
    
            if extra:
                self.extra_gates.append(m)
            return m

    def CNOT(self, control: int, target: int, extra=True):
            m = super().CNOT(control, target, add_to_circuit=False)
    
            if extra:
                self.extra_gates.append(m)
            return m


    def create_three_bit_code(self):
        gs = []

        # Encoding
        # gs.append(self.H(0, add_to_circuit=False))
        gs.append(self.CNOT(0, 1, extra=False))
        gs.append(self.CNOT(0, 2, extra=False))

        return gs

    def add_stabilizers(self):
        # Adding stabilizers
        s1 = []
        s1.append(self.CNOT(0, 3, extra=False))
        s1.append(self.CNOT(1, 3, extra=False))

        s2 = []
        s2.append(self.CNOT(1, 4, extra=False))
        s2.append(self.CNOT(2, 4, extra=False))

        return s1, s2

    def measure_stabilizer(self, state, bit):
        return self._measure_quick(state, bit)

    # Not used
    def add_X_noise(self, prob):
        if self.rng.random() < prob:
            bit = self.rng.integers(self.distance)
            self.X(bit)

        return

    def run(
            self,
            physical_error_prob:float=0.2,
            error_weight: int=1,
            measurement_error_prob: float=0.2,
            shots: int=10,
            rounds: int = 25,
            random_initial_state: bool=False,
        ):

        result = 0

        self.matching_graph = self._build_matching_graph(
            rounds=rounds,
            px=physical_error_prob,
            pm=measurement_error_prob
        )


        for _ in range(shots):

            # TODO: Make sure this is copying correctly
            state = self.initial_state.copy()

            if random_initial_state:
                state = self.CNOT(0, 1, extra=False) @ self.CNOT(0, 2, extra=False) @ self.H(0, extra=False) @ state
                intial_result, state = self._measure_quick(state, 0)


            syndrome_measurements = []
            true_physical_error = []
            true_measurement_error = []
            # syndrome, state, phys_error, meas_error = self.run_round(
            #     state=state,
            #     error_prob=physical_error_prob,
            #     error_weight=error_weight,
            #     measurement_noise_prob=measurement_error_prob,
            #     first_round=True,
            # )

            # syndrome_measurements.append(syndrome)
            # true_physical_error.append(phys_error)
            # true_measurement_error.append(meas_error)

            for _ in range(rounds):
                syndrome, state, phys_error, meas_error = self.run_round(
                    state=state,
                    error_prob=physical_error_prob,
                    error_weight=error_weight,
                    measurement_noise_prob=measurement_error_prob,
                )

                syndrome_measurements.append(syndrome)
                true_physical_error.append(phys_error)
                true_measurement_error.append(meas_error)


            detections = self.detect_changes(syndrome_measurements)
            matches = self.match_errors(self.matching_graph, detections)
            predicted_error = self.extract_predicted_errors(matches)

            if self.verbose:
                print('Syndrome:', syndrome_measurements)
                print('\n')
                print('Physical Errors:', true_physical_error)
                print('\n')
                print('Measurement Errors:', true_measurement_error)

                self.display_syndrome(detections)
                print(matches)

            true_total_error = np.sum(true_physical_error, axis=0) % 2

            diff = sum(np.abs(true_total_error - predicted_error))

            if diff != 0:
                result += 1

            # print(diff, true_total_error, predicted_error)

        return result/shots

    def run_round(
            self,
            state,
            error_prob: float,
            error_weight: int,
            measurement_noise_prob: float,
            first_round: bool=False,
        ):

        error = []
        unaffected_bits = list(range(self.distance))

        # For single error per run per bit
        true_phys_error = [0, 0, 0]
        for i in range(error_weight):
            if self.rng.random() < error_prob:
                idx = self.rng.integers(len(unaffected_bits))
                bit = unaffected_bits.pop(idx)
                true_phys_error[bit] = 1
                error.append(self.X(bit, extra=False))

        if not error:
            error = [np.diag(np.ones(2**(self.distance + 2)))]

        # Stabilizer noise
        stab_error = []
        true_meas_error = [0, 0]
        for i in range(2):
            if self.rng.random() < measurement_noise_prob:
                true_meas_error[i] = 1
                stab_error.append(self.X(3 + i, extra=False))

        if not stab_error:
            stab_error = [np.diag(np.ones(2**(self.distance + 2)))]

        # if first_round:
        #     gates = np.concatenate([self.code_state, error, self.s1, self.s2, stab_error])
        # else:
        gates = np.concatenate([error, self.s1, self.s2, stab_error])

        new_state = self.run_circuit(gates, state)

        # print('###', bin(np.argmax(new_state)))
        m1, ns1 = self.measure_stabilizer(new_state, 3)
        # print('####', bin(np.argmax(ns1)))
        m2, ns2 = self.measure_stabilizer(new_state, 4)

        # print(m1, m2)

        ns2 = self.reset_stabilizers(ns2)
        
        syndrome = [m1, m2]

        return syndrome, ns2, true_phys_error, true_meas_error

    def run_circuit(self, gates, initial_state):
        # print('#', bin(np.argmax(initial_state)))
        for gate in gates:
            # print(len(gates))
            initial_state = gate @ initial_state
        # print('##', bin(np.argmax(initial_state)))
        return initial_state

    def reset_stabilizers(self, state):
        new_state = np.zeros(len(state))

        for i in range(len(state)):
            if state[i][0] != 0:
                new_component = (i >> 2) << 2
                new_state[new_component] = state[i][0]

        new_state = new_state.reshape(-1, 1)

        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state = new_state / norm

        return new_state

    def display_syndrome(self, syndrome):
        full_syn = []
        for meas in syndrome:
            rnd = ['_' if s == 0 else 'e' for s in meas]
            rnd.append('\n')
            full_syn.append(''.join(rnd))

        print(''.join(full_syn))


    def detect_changes(self, syndrome: list) -> list:
        syndrome = np.array(syndrome)

        syndrome_change = np.abs(np.diff(syndrome, axis=0, prepend=np.zeros((1, syndrome.shape[1]))))

        return syndrome_change
    

    def _build_matching_graph(self, rounds, px, pm):
        matching = Matching()

        if px > 0:
            weightx = -np.log(px/(1-px))
        if pm > 0:
            weightm = -np.log(pm/(1-pm))


        for i in range(rounds):
            # Adding physical edges
            if px > 0:
                matching.add_boundary_edge(2*i, weight=weightx)
                matching.add_edge(2*i, 2*i+1, weight=weightx)
                matching.add_boundary_edge(2*i+1, weight=weightx)

            # Adding measurement edges (currently no time boundry TODO)
            if pm > 0:
                if i < rounds - 1:
                    matching.add_edge(2*i, 2*(i + 1), weight=weightm)
                    matching.add_edge(2*i + 1, 2*(i + 1) + 1, weight=weightm)

        return matching


    def match_errors(self, matching_graph: Matching, detection_events) -> list:
        '''
        Instead of looking at looking at the last n time steps, we can look at whole space time 'grid'
        Put detections of the vertices of the grid
        grid also has legs on the edges connecting to nothing
        physical errors give either one or two detection events. These events only last for one timestep (qubits should remain in same state if no error)
        measurement errors give two detection events, spread accross two timesteps (ancillas are reset when errors occur)
        if two errors happen on the same stabilizer (measurement error and one of its physical qubits flips, or both physical qubits flip ) it cancels out in the syndrome
        detection errors only show if they have an odd number of errors associated to them
        assume each detection error (vertices) only has one error associated with it (mathematically more likely)
        find a matching between detection errors (following grid lines) 
        this includes between points along an edge which connects nothing
        prob of a given edge is given by px or pm (probs of physical or matching errors)
        if a match goes accross more than one edge, we multiple probabilities of both edges to get total prob
        want to find matching with highest probability

        IDEAS
        if zero for more than k turns, 'cut' the graph. Assume no cumulative measurement errors

        Option 1: MILP
        Option 2: Blossom algorithm
            - Change/wrap Nx
            - Pymatching  
        Option 3: Genetic algorithm
        Option 4: RL/ML
        Option 5: Brute force
        '''

        matched = matching_graph.decode_to_matched_dets_array(detection_events.flatten())
        
        return matched


    def extract_predicted_errors(self, matched_errors):
        errors = np.zeros(3)
        for match in matched_errors:
            # TODO. This works specifically for 3 bit repitition code with bitflip noise
            if -1 not in match:
                s1, t1 = match[0] % 2, match[0] // 2
                s2, t2 = match[1] % 2, match[1] // 2

                if s1 - s2 != 0:
                    errors[1] += 1
            else:
                for i in match:
                    if i != -1:
                        s = i % 2

                        if s == 0:
                            errors[0] += 1
                        else:
                            errors[2] += 1

        return errors % 2
                    


# code = RepititionCode(3, verbose=False)
# error_rate = code.run(
#     physical_error_prob=0.04,
#     error_weight=3,
#     measurement_error_prob=0.04,
#     shots=100,
#     rounds=25,
#     random_initial_state=True
# )

# print(f'Error Rate: {error_rate}')