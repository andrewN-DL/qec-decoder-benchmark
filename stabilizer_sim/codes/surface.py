import numpy as np

import operators as o
from codes.base_code import Code
import decoding.utils as d
import decoding.toric_decoding as to

import time

# TODO: At the moment always square
class SurfaceCode(Code):
    def __init__(self, dist, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.dist = dist
        self.initial_state_h = o.Operator([self.dist + 1, self.dist + 2])
        self.initial_state_v = o.Operator([self.dist, self.dist + 1])

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

        state_v = self.initial_state_v.copy()
        state_h = self.initial_state_h.copy()
        
        face_syndrome = []
        vert_syndrome = []

        # Generating noise
        x_noise_full_v = self.rng.random((rounds, self.dist, self.dist+1))
        y_noise_full_v = self.rng.random((rounds, self.dist, self.dist+1))
        z_noise_full_v = self.rng.random((rounds, self.dist, self.dist+1))

        x_noise_full_h = self.rng.random((rounds, self.dist+1, self.dist+2))
        y_noise_full_h = self.rng.random((rounds, self.dist+1, self.dist+2))
        z_noise_full_h = self.rng.random((rounds, self.dist+1, self.dist+2))

        face_meas_noise_full = self.rng.random((rounds, self.dist, self.dist+2))
        vert_meas_noise_full = self.rng.random((rounds, self.dist+1, self.dist+1))

        op_dim_v, op_dim_h = [self.dist, self.dist + 1], [self.dist + 1, self.dist + 2]
        for rnd in range(rounds):

            x_noise_v = np.argwhere(x_noise_full_v[rnd] < px)
            y_noise_v = np.argwhere(y_noise_full_v[rnd] < py)
            z_noise_v = np.argwhere(z_noise_full_v[rnd] < pz)

            x_noise_h = np.argwhere(x_noise_full_h[rnd] < px)
            y_noise_h = np.argwhere(y_noise_full_h[rnd] < py)
            z_noise_h = np.argwhere(z_noise_full_h[rnd] < pz)

            o1v, o2v, o3v = o.X(op_dim_v, x_noise_v), o.Y(op_dim_v, y_noise_v), o.Z(op_dim_v, z_noise_v)
            o1h, o2h, o3h = o.X(op_dim_h, x_noise_h), o.Y(op_dim_h, y_noise_h), o.Z(op_dim_h, z_noise_h)
            
            state_v = state_v * o1v * o2v * o3v
            state_h = state_h * o1h * o2h * o3h

            # print('##', y_noise_v, z_noise_v, '###')
            # print(o1v, o2v, o3v)

            # print('ST', state_v)
            face_syndrome_t, vert_syndrome_t = self.syndromes(state_v.ex, state_v.ez, state_h.ex, state_h.ez)
            
            face_syndrome_t = face_syndrome_t.astype(int) ^ (face_meas_noise_full[rnd] < pm)
            vert_syndrome_t = vert_syndrome_t.astype(int) ^ (vert_meas_noise_full[rnd] < pm)

            face_syndrome.append(face_syndrome_t)
            vert_syndrome.append(vert_syndrome_t)


        face_syndrome_t, vert_syndrome_t = self.syndromes(state_v.ex, state_v.ez, state_h.ex, state_h.ez)

        face_syndrome.append(face_syndrome_t.astype(int))
        vert_syndrome.append(vert_syndrome_t.astype(int))

        face_detections, vert_detections = d.detect_changes(face_syndrome), d.detect_changes(vert_syndrome)

        t = time.time()
        try:
            matches = d.match_errors(matching_graph, np.array([face_detections, vert_detections]))
        except Exception as e:
            print('Face\n', face_detections, '\n\nVertex \n' ,vert_detections, '\n\n')
            raise e
        self.times['matching'].append(time.time() - t)

        t = time.time()
        predicted_errors = to.extract_toric_predicted_errors(matches, self.dist, rounds)
        self.times['pred_error'].append(time.time() - t)
        
        if self.verbose:
            print('Z\n', face_detections, '\n\nX \n' ,  vert_detections, '\n\n')
            print(matches)
            print('True Errors:\n',state_v, state_h)
            for i in range(3):
                print('  ',predicted_errors[0][2*i],'\n',  predicted_errors[0][2*i + 1])
            print('\n\n')
            for i in range(3):
                print('  ',predicted_errors[1][2*i],'\n',  predicted_errors[1][2*i + 1])

        t = time.time()
        # error = self.check_error(state.ex, state.ez, predicted_errors)
        # self.times['check_error'].append(time.time() - t)

        # return error, state, predicted_errors

    # TODO: Check naming. Okay for now due to noise symmetry but should check
    def syndromes(self, exv, ezv, exh, ezh):
        
        syn_face = np.zeros([self.dist, self.dist+2])
        # Rough boundaries
        syn_face[:, 0] = ezh[:-1, 0] ^ np.roll(ezh, -1, axis=0)[:-1, 0] ^ ezv [:, 0]
        syn_face[:, -1] = ezh[:-1, -1] ^ np.roll(ezh, -1, axis=0)[:-1, -1] ^ ezv [:, -1]
        # Bulk
        syn_face[:, 1:-1] = ezh[:-1, 1:-1] ^ np.roll(ezh, -1, axis=0)[:-1, 1:-1] ^ ezv[:, :-1] ^ np.roll(ezv, -1, axis=1)[:, :-1]


        syn_vert = np.zeros([self.dist+1, self.dist+1])
        # Smooth boundaries
        syn_vert[0, :] = exh[0, :-1] ^ np.roll(exh, -1, axis=0)[0, :-1] ^ exv [0, :]
        syn_vert[-1, :] = exh[-1, :-1] ^ np.roll(exh, -1, axis=0)[-1, :-1] ^ exv [-1, :]
        # Bulk
        syn_vert[1:-1, :] = exh[1:-1, :-1] ^ np.roll(exh, -1, axis=0)[1:-1, :-1] ^ exv[:-1, :] ^ np.roll(exv, -1, axis=1)[:-1, :]


        # print(syn_face, syn_vert)

        # # Z-error array
        # hz, vz = ez[0::2], ez[1::2]          
        # syn_face = hz ^ vz ^ np.roll(vz, -1, axis=1) ^ np.roll(hz, -1, axis=0)

        # # X-error array
        # hx, vx = ex[0::2], ex[1::2]
        # syn_vert = hx ^ np.roll(hx, 1, axis=1) ^ vx ^ np.roll(vx, 1, axis=0)
        # return syn_face, syn_vert

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
