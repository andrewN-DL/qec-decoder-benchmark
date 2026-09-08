from circuit import Circuit
import numpy as np
import copy

class RepititionCode(Circuit):
    def __init__(self, distance):
        super().__init__(distance + 2)
        self.distance = distance

        self.initial_state = copy.deepcopy(self.state)

        self.decoding_map = {
            '10': 0,
            '11': 1,
            '01': 2
        }

        if distance == 3:
            self.code_state = self.create_three_bit_code()
            self.s1, self.s2 = self.add_stabilizers()


    def create_three_bit_code(self):
        dim = 5

        gs = []

        # Encoding
        gs.append(self.H(0, add_to_circuit=False))
        gs.append(self.CNOT(0, 1, add_to_circuit=False))
        gs.append(self.CNOT(0, 2, add_to_circuit=False))

        return gs

    def add_stabilizers(self):
        # Adding stabilizers
        s1 = []
        s1.append(self.CNOT(0, 3, add_to_circuit=False))
        s1.append(self.CNOT(1, 3, add_to_circuit=False))

        s2 = []
        s2.append(self.CNOT(1, 4, add_to_circuit=False))
        s2.append(self.CNOT(2, 4, add_to_circuit=False))

        return s1, s2

    def measure_stabilizer(self, state, bit):
        return self._measure_quick(state, bit)

    def add_X_noise(self, prob):
        if np.random.random() < prob:
            bit = np.random.randint(self.distance)
            self.X(bit)

        return

    def run(self, noise: float=0.2, shots: int=10):
        state = copy.deepcopy(self.initial_state)

        for _ in range(shots):
            error = [np.diag(np.ones(2**(self.distance + 2)))]
            if np.random.random() < noise:
                bit = np.random.randint(3)
                error = [self.X(bit)]
                print(f'Error on bit {bit}')
            gates = np.concatenate([self.code_state, error, self.s1, self.s2])
            new_state = self.run_circuit(gates, state)

            m1, ns1 = self.measure_stabilizer(new_state, 3)
            m2, ns2 = self.measure_stabilizer(new_state, 4)

            syndrome = [m1, m2]

            print(f'Syndrome: {syndrome}')

            final_state = self.decode(''.join([str(i) for i in syndrome]), ns2)

            print(self._measure_all(final_state))

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
        return self.X(correct_bit) @ state
