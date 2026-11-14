import numpy as np

import operators as o
from codes.base_code import Code
import decoding.utils as d
import decoding.toric_decoding as to

import time


class ToricCode(Code):
    def __init__(self, dist, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.dist = dist
        self.initial_state = o.Operator([2*self.dist, self.dist])

    # TODO: This can prob go in Code class
    def run(
            self,
            shots,
            rounds,
            px,
            py,
            pz,
            pm,
            show_errors=False,
    ):

        matching_graph = to.build_toric_matching_graph(self.dist, rounds, px, py, pz, pm)

        # Checking which part of the process is slowing down the run
        self.times = {
            'matching': [],
            'det_conv': [],
            'get_synd': [],
            'noise_gen': [],
            'pred_error': [],
            'check_error': [],
            'shot': [],
            'mult_ops': [],
            'measure_stabs': []
        }

        result = {
            'no_error': 0,
            'logical': 0,
            'outside': 0
        }

        for _ in range(shots):
            st = time.time()
            shot_res = self.run_round(rounds, px, py, pz, pm, matching_graph)
            result[shot_res[0][0]] += 1
            self.times['shot'].append(time.time() - st)

        # for k, v in self.times.items():
        #     print(k, np.mean(v))

        # Return last error/prediction
        if show_errors:
            return shot_res 

        return {k: v/shots for k, v in result.items()}

    def run_round(
        self,
        rounds,
        px,
        py,
        pz,
        pm,
        matching_graph
    ):

        state = self.initial_state.copy()
        
        x_syndrome = []
        z_syndrome = []

        # Generating noise
        t = time.time()
        x_noise_full = self.rng.random((rounds, 2*self.dist, self.dist))
        y_noise_full = self.rng.random((rounds, 2*self.dist, self.dist))
        z_noise_full = self.rng.random((rounds, 2*self.dist, self.dist))

        x_meas_noise_full = self.rng.random((rounds, self.dist, self.dist))
        z_meas_noise_full = self.rng.random((rounds, self.dist, self.dist))
        self.times['noise_gen'].append(time.time() - t)

        t = time.time()


        op_dim = [2* self.dist, self.dist]
        for rnd in range(rounds):

            x_noise = np.argwhere(x_noise_full[rnd] < px)
            y_noise = np.argwhere(y_noise_full[rnd] < py)
            z_noise = np.argwhere(z_noise_full[rnd] < pz)

            t1 = time.time()
            o1, o2, o3 = o.X(op_dim, x_noise), o.Y(op_dim, y_noise), o.Z(op_dim, z_noise)
            
            state = state * o1 * o2 * o3
            self.times['mult_ops'].append(time.time() - t1)

            z_syndrome_t, x_syndrome_t = self.syndromes(state.ex, state.ez)
            
            x_syndrome_t = x_syndrome_t.astype(int) ^ (x_meas_noise_full[rnd] < pm)
            z_syndrome_t = z_syndrome_t.astype(int) ^ (z_meas_noise_full[rnd] < pm)

            x_syndrome.append(x_syndrome_t)
            z_syndrome.append(z_syndrome_t)


        z_syndrome_t, x_syndrome_t = self.syndromes(state.ex, state.ez)

        x_syndrome.append(x_syndrome_t.astype(int))
        z_syndrome.append(z_syndrome_t.astype(int))

        self.times['get_synd'].append(time.time() - t)

        t = time.time()
        x_detections, z_detections = d.detect_changes(x_syndrome), d.detect_changes(z_syndrome)
        self.times['det_conv'].append(time.time() - t)

        t = time.time()
        try:
            print(np.array([x_detections, z_detections]).flatten(), type(np.array([x_detections, z_detections]).flatten()))
            matches = d.match_errors(matching_graph, np.array([x_detections, z_detections]))
        except Exception as e:
            print('Z\n', x_detections, '\n\nX \n' ,z_detections, '\n\n')
            print(x_meas_noise_full)
            raise e
        self.times['matching'].append(time.time() - t)

        t = time.time()
        predicted_errors = to.extract_toric_predicted_errors(matches, self.dist, rounds)
        self.times['pred_error'].append(time.time() - t)
        
        if self.verbose:
            print('Z\n', x_detections, '\n\nX \n' ,z_detections, '\n\n')
            print(matches)
            print('True Errors:\n',state)
            for i in range(3):
                print('  ',predicted_errors[0][2*i],'\n',  predicted_errors[0][2*i + 1])
            print('\n\n')
            for i in range(3):
                print('  ',predicted_errors[1][2*i],'\n',  predicted_errors[1][2*i + 1])

        t = time.time()
        error = self.check_error(state.ex, state.ez, predicted_errors)
        self.times['check_error'].append(time.time() - t)

        return error, state, predicted_errors

    def syndromes(self, ex, ez):
        # Z-error array
        hz, vz = ez[0::2], ez[1::2]          
        syn_face = hz ^ vz ^ np.roll(vz, -1, axis=1) ^ np.roll(hz, -1, axis=0)

        # X-error array
        hx, vx = ex[0::2], ex[1::2]
        syn_vert = hx ^ np.roll(hx, 1, axis=1) ^ vx ^ np.roll(vx, 1, axis=0)
        return syn_face, syn_vert
    

    def check_error(
            self,
            true_x_errors,
            true_z_errors,
            predicted_errors
    ):

        '''
        Starting with X errors. Et = true errors, Ep = predicted errors
        EtEp is always a continuious loop. S = set of faces of graph (element of stabilizer group)
        If EtEp in dS (boundaries of S) then no logical error
        Else logical operation (in N/S)
        If EtEp in dS, the loop will pass a given row/col an even number of times
        Otherwise it will be an odd number of times

        For Z errors (dual lattice) We can ecamine horizontal and vertical lines in a similar way
        '''
        is_error = 0

        x_errors = predicted_errors[0]
        combined_x = np.array(x_errors ^ true_x_errors)

        # Checking if outside Normalizer
        # TODO: differentiate between logical error and outside code space
        def meas_error_x(dist, state, i, j):
            relevant_data_bits = [
                state[2*i, j],
                state[2*i, (j-1)%dist],
                state[(2*i+1)%(2*dist), j],
                state[(2*i-1) % (2*dist), j],
                ]

            parity = sum(relevant_data_bits) % 2

            return parity, relevant_data_bits
        
        for i1 in range(self.dist):
            for j1 in range(self.dist):
                r = meas_error_x(self.dist, combined_x, i1, j1)
                if r[0] == 1:
                    return 'outside', [], []

        z_errors = predicted_errors[1]
        combined_z = np.array(z_errors ^ true_z_errors)

        # Checking if outside Normalizer
        # TODO: differentiate between logical error and outside code space
        def meas_error_z(dist, state, i, j):
            relevant_data_bits = [
                state[(2*i+1)%(2*dist), j],
                state[(2*i+1)%(2*dist), (j+1)%dist],
                state[2*i, j],
                state[(2*i+2) % (2*dist), j],
                ]

            parity = sum(relevant_data_bits) % 2

            return parity, relevant_data_bits
        
        for i in range(self.dist):
            for j in range(self.dist):
                r = meas_error_z(self.dist, combined_z, i, j)
                if r[0] == 1:
                    return 'outside', [], []

        errors_x = [0, 0]

        # Checking one row
        if sum(combined_x[1, :]) % 2 == 1:
            errors_x[0] = 1 
            is_error = 1

        # Checking one column
        h_vals = [combined_x[2*v, 0] for v in range(self.dist)]
        if sum(h_vals) % 2 == 1:
            errors_x[1] = 1
            is_error = 1

        # Doing similar on dual graph (rows <-> columns)
        errors_z = [0, 0]
        if sum(combined_z[0, :]) % 2 == 1:
            errors_z[0] = 1
            is_error = 1

        v_vals = [combined_z[2*v + 1, 0] for v in range(self.dist)]
        if sum(v_vals) % 2 == 1:
            errors_z[1] = 1
            is_error = 1

        if is_error:
            return 'logical', errors_x, errors_z

        return 'no_error', errors_x, errors_z
