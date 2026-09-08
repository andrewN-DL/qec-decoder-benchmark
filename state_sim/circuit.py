import numpy as np
import copy

class Circuit:

    def __init__(self, dim: int):
        self.R2 = 1 / np.sqrt(2)
        self.dim = dim

        self.state = np.array([[1]] + [[0]] * (2**dim - 1))

        self.gates = []
        

    def H(self, bit: int, add_to_circuit: bool=True):

        assert bit >= 0

        had = np.array([
            [1*self.R2, 1*self.R2],
            [1*self.R2, -1*self.R2]])

        id = np.array([
            [1, 0],
            [0, 1]
        ])

        if self.dim == 1:
            return had

        chain = np.zeros(self.dim)
        chain[bit] = 1

        m0 = had if chain[0] == 1 else id
        m1 = had if chain[1] == 1 else id
        prod = np.kron(m0, m1)

        if self.dim == 2:
            return prod
        
        for i in chain[2:]:
            m = had if i == 1 else id
            prod = np.kron(prod, m)

        if add_to_circuit:
            self.gates.append(prod)
        
        return prod

    def Z(self, bits: list, add_to_circuit: bool=True):

        z = np.diag([1, -1])
        id = np.diag([1, 1])

        if self.dim == 1:
            return z

        chain = np.zeros(self.dim)
        chain[bits] = 1

        m0 = z if chain[0] == 1 else id
        m1 = z if chain[1] == 1 else id
        prod = np.kron(m0, m1)

        if self.dim == 2:
            return prod
        
        for i in chain[2:]:
            m = z if i == 1 else id
            prod = np.kron(prod, m)

        if add_to_circuit:
            self.gates.append(prod)

        return prod

    def X(self, bits: list, add_to_circuit: bool=True):

        x = np.array([
            [0, 1],
            [1, 0]
        ])
        id = np.diag([1, 1])

        if self.dim == 1:
            return x

        chain = np.zeros(self.dim)
        chain[bits] = 1

        m0 = x if chain[0] == 1 else id
        m1 = x if chain[1] == 1 else id
        prod = np.kron(m0, m1)

        if self.dim == 2:
            return prod
        
        for i in chain[2:]:
            m = x if i == 1 else id
            prod = np.kron(prod, m)

        if add_to_circuit:
            self.gates.append(prod)

        return prod

    def CNOT(self, control, target, add_to_circuit: bool=True):
        x = np.array(
                [[0, 1],
                [1, 0]]
            )
        id = np.diag([1, 1])

        m = np.zeros([2**self.dim, 2**self.dim])

        for state in range(2**self.dim):
            binary = np.zeros(self.dim)
            for i, digit in enumerate(reversed(bin(state)[2:])):
                binary[-(i + 1)] = digit

            binary[target] = (binary[control] + binary[target]) % 2

            num = 0
            for i, comp in enumerate(reversed(binary)):
                num += (comp * (2**i))

            m[state, int(num)] = 1

        if add_to_circuit:
            self.gates.append(m)

        return m

    def bin_state(self, i, dim=None, display=True):
        dim = dim or self.dim

        binary = np.zeros(dim)
        for i, digit in enumerate(reversed(bin(i)[2:])):
            binary[-(i + 1)] = digit

        binary = [str(int(n)) for n in binary]
        if display:
            return '|' + ''.join(binary) + '⟩'
        
        return ''.join(binary)

    def measure(self, qubit):
        self.gates.append(qubit)

    def _measure_quick(self, state, index):

        # Sampling basis state in Z-basis
        sample = np.random.choice(len(state), p=(state**2).reshape(1, -1)[0])

        # Measuring relevant bit
        result = self.bin_state(sample, self.dim, display=False)[index]

        # Projecting onto post measurement space
        state = np.array([s if (self.bin_state(idx, self.dim, display=False)[index] == result) else np.array([0.0]) for idx, s in enumerate(state)]).reshape(-1, 1)

        # Renormalising
        state = state / np.sum(state**2)

        return int(result), state


    def _measure_all(self, state):
    
            # Sampling basis state in Z-basis
            sample = np.random.choice(len(state), p=(state**2).reshape(1, -1)[0])
    
            # Measuring relevant bit
            result = self.bin_state(sample, self.dim, display=True)
    
            return result

    # def compile(self, shots: int=10):
    #     print('Running')
    #     for gate in gates:
    #         if isinstance(gate, int):
    #             res, ss = self._measure_quick(ss, gate)
    #             print(f'Measured {gate}. Result:', res)
    #         else:
    #             ss = gate @ ss
        

    def run(self, noise: bool=False, shots: int=10):

        for _ in range(shots):
            print('Running')
            gates = copy.deepcopy(self.gates)
            if noise:
                if np.random.random() < 0.2:
                    bit = np.random.randint(3)
                    gates.insert(3, self.X(bit))
                    print('Hit', bit)
            ss = copy.deepcopy(self.state)
            for gate in gates:
                if isinstance(gate, int):
                    res, ss = self._measure_quick(ss, gate)
                    print(f'Measured {gate}. Result:', res)
                else:
                    ss = gate @ ss

            print('Final Measurement:', self._measure_all(ss))

        return self.state
