from circuit import Circuit
import numpy as np
import copy

# currently only works for 3-qubit code
class RepititionCode(Circuit):
    def __init__(self, distance: int):
        super().__init__(distance + 2)
        self.distance = distance

        self.initial_state = copy.deepcopy(self.state)

        # TODO: This works for 3 qubit code. For say 9 qubit this would prob need to be programmatically. Shouldn't be too hard
        self.decoding_map = {
            '10': 0,
            '11': 1,
            '01': 2
        }

        if distance == 3:
            self.code_state = self.create_three_bit_code()
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

    def add_X_noise(self, prob):
        if np.random.random() < prob:
            bit = np.random.randint(self.distance)
            self.X(bit)

        return

    def run(self, error_prob: float=0.2, error_weight: int=1, shots: int=10, random_initial_state: bool=False):
        result = []

        state = copy.deepcopy(self.initial_state)

        for _ in range(shots):

            if random_initial_state:
                state = self.H(0, extra=False) @ state
                intial_result, state = self._measure_quick(state, 0)
    
            # print(state)
            error = []
            unaffected_bits = list(range(self.distance))

            # For single error per run
            for _ in range(error_weight):
                if np.random.random() < error_prob:
                    # print(unaffected_bits)
                    idx = np.random.randint(len(unaffected_bits))
                    bit = unaffected_bits.pop(idx)
                    error.append(self.X(bit, extra=False))
                    # print(f'Error on bit {bit}')

            if not error:
                error = [np.diag(np.ones(2**(self.distance + 2)))]

            gates = np.concatenate([self.code_state, error, self.s1, self.s2])
            new_state = self.run_circuit(gates, state)

            m1, ns1 = self.measure_stabilizer(new_state, 3)
            m2, ns2 = self.measure_stabilizer(new_state, 4)

            syndrome = [m1, m2]

            # print(f'Syndrome: {syndrome}')

            final_state = self.decode(''.join([str(i) for i in syndrome]), ns2)

            final_state = self._measure_all(final_state)

            result.append((intial_result) == int(final_state[1]))

            # print(f'Initial logical state: {intial_result}, Final result:  {final_state[:4] + final_state[6:]}. Decoding successful: {(intial_result) == int(final_state[1])}')

        logical_error_rate = 1 - (np.sum(result))/len(result)
        # print(f'Total logical error: {1 - (np.sum(result))/len(result)}')

        return logical_error_rate

    def run_circuit(self, gates, initial_state):
        
        for gate in gates:
            initial_state = gate @ initial_state

        return initial_state

    def decode(self, syndrome: list, state):
        # print(syndrome, self.decoding_map.values())
        if not syndrome in self.decoding_map.keys():
            # print('No error')
            return state
        
        correct_bit = self.decoding_map[syndrome]
        # print(correct_bit)
        return self.X(correct_bit, extra=False) @ state


code = RepititionCode(3)
code.run(error_prob=0.1, error_weight=3, shots=2000, random_initial_state=True)