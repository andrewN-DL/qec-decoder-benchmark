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

    # Not used
    def add_X_noise(self, prob):
        if np.random.random() < prob:
            bit = np.random.randint(self.distance)
            self.X(bit)

        return

    def run(
            self,
            error_prob:float=0.2,
            error_weight: int=1,
            measurement_noise_prob: float=0.2,
            shots: int=10,
            rounds: int = 25,
            random_initial_state: bool=False
        ):

        result = []

        # TODO: Make sure this is copying correctly
        state = self.initial_state.copy()

        for _ in range(shots):
            if random_initial_state:
                state = self.H(0, extra=False) @ state
                intial_result, state = self._measure_quick(state, 0)

            syndrome_measurements = []
            true_physical_error = []
            true_measurement_error = []
            syndrome, state, phys_error, meas_error = self.run_round(
                state=state,
                error_prob=error_prob,
                error_weight=error_weight,
                measurement_noise_prob=measurement_noise_prob,
                first_round=True,
            )

            syndrome_measurements.append(syndrome)
            true_physical_error.append(phys_error)
            true_measurement_error.append(meas_error)

            for _ in range(rounds):
                syndrome, state, phys_error, meas_error = self.run_round(
                    state=state,
                    error_prob=error_prob,
                    error_weight=error_weight,
                    measurement_noise_prob=measurement_noise_prob,
                )

                syndrome_measurements.append(syndrome)
                true_physical_error.append(phys_error)
                true_measurement_error.append(meas_error)

            print('Syndrome:', syndrome_measurements)
            print('\n')
            print('Physical Errors:', true_physical_error)
            print('\n')
            print('Measurement Errors:', true_measurement_error)

            self.display_syndrome(syndrome_measurements)

            final_state = self.decode(''.join([str(i) for i in syndrome]), state)

            final_state = self._measure_all(final_state)

            result.append((intial_result) == int(final_state[1]))

        logical_error_rate = 1 - (np.sum(result))/len(result)

        return logical_error_rate

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
            if np.random.random() < error_prob:
                idx = np.random.randint(len(unaffected_bits))
                bit = unaffected_bits.pop(idx)
                true_phys_error[bit] = 1
                error.append(self.X(bit, extra=False))

        if not error:
            error = [np.diag(np.ones(2**(self.distance + 2)))]

        # Stabilizer noise
        stab_error = []
        true_meas_error = [0, 0]
        for i in range(2):
            if np.random.random() < measurement_noise_prob:
                true_meas_error[i] = 1
                stab_error.append(self.X(3 + i, extra=False))

        if not stab_error:
            stab_error = [np.diag(np.ones(2**(self.distance + 2)))]

        if first_round:
            gates = np.concatenate([self.code_state, error, self.s1, self.s2, stab_error])
        else:
            gates = np.concatenate([error, self.s1, self.s2, stab_error])

        new_state = self.run_circuit(gates, state)

        m1, ns1 = self.measure_stabilizer(new_state, 3)
        m2, ns2 = self.measure_stabilizer(new_state, 4)

        ns2 = self.reset_stabilizers(ns2)
        
        syndrome = [m1, m2]

        return syndrome, ns2, true_phys_error, true_meas_error

    def run_circuit(self, gates, initial_state):
        
        for gate in gates:
            initial_state = gate @ initial_state

        return initial_state

    def reset_stabilizers(self, state):
        new_state = np.zeros(len(state))

        for i in range(len(state)):
            if state[i][0] != 0:
                new_component = (i >> 2) << 2
                new_state[new_component] = state[i][0]

        new_state = new_state.reshape(-1, 1)

        return new_state

    def display_syndrome(self, syndrome):
        full_syn = []
        for meas in syndrome:
            rnd = ['_' if s == 0 else 'e' for s in meas]
            rnd.append('\n')
            full_syn.append(''.join(rnd))

        print(''.join(full_syn))


    def decode(self, syndrome: list, state):
        # print(syndrome, self.decoding_map.values())
        if not syndrome in self.decoding_map.keys():
            # print('No error')
            return state
        
        correct_bit = self.decoding_map[syndrome]
        # print(correct_bit)
        return self.X(correct_bit, extra=False) @ state


code = RepititionCode(3)
code.run(
    error_prob=0.04,
    error_weight=3,
    measurement_noise_prob=0.04,
    shots=1,
    rounds=50,
    random_initial_state=True
)