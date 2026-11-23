import numpy as np

import operators as o
from codes.base_code import Code
import decoding.utils as d
import decoding.surface_decoding as so

import time

# TODO: At the moment always square
class SurfaceCode(Code):
    def __init__(self, dist, verbose: bool=False, seed: int=None):
        super().__init__(verbose, seed)

        self.dist = dist
        self.initial_state_h = o.Operator([self.dist, self.dist])
        self.initial_state_v = o.Operator([self.dist - 1, self.dist - 1])

    # TODO: This can prob go in Code class
    def run(
            self,
            shots,
            rounds,
            true_noise,
            assumed_noise,
            show_errors=False,
    ):
        pxt, pyt, pzt, pmt = tuple(true_noise)
        pxa, pya, pza, pma = tuple(assumed_noise)

        matching_graph = so.build_matching_graph(self.dist, rounds, pxa, pya, pza, pma)

        # TODO: Implement or remove (Test)
        result = 0
        errors = [0, 0]

        for _ in range(shots):
            shot_res = self.run_shot(rounds, pxt, pyt, pzt, pmt, matching_graph)

            result += shot_res[0][0]
            errors[0] += shot_res[0][1][0]
            errors[1] += shot_res[0][1][1]

        # Return last error/prediction
        if show_errors:
            return shot_res 

        return result/shots, [k/shots for k in errors]

    def run_shot(
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
        noise_full_v, noise_full_h, face_meas_noise_full, vert_meas_noise_full = self.generate_noise(rounds)

        op_dim_v, op_dim_h = [self.dist - 1, self.dist - 1], [self.dist, self.dist]
        for rnd in range(rounds):

            face_syndrome_t, vert_syndrome_t, state_v, state_h = self.run_round(
                                                    state_v,
                                                    state_h,
                                                    op_dim_v,
                                                    op_dim_h,
                                                    noise_full_v[rnd],
                                                    noise_full_h[rnd],
                                                    face_meas_noise_full[rnd],
                                                    vert_meas_noise_full[rnd],
                                                    px,
                                                    py,
                                                    pz,
                                                    pm,
                                                )

            face_syndrome.append(face_syndrome_t)
            vert_syndrome.append(vert_syndrome_t)

        # Final round without measurement errors
        face_syndrome_t, vert_syndrome_t = self.get_syndromes(state_v.ex, state_v.ez, state_h.ex, state_h.ez)

        face_syndrome.append(face_syndrome_t.astype(int))
        vert_syndrome.append(vert_syndrome_t.astype(int))

        face_detections, vert_detections = d.detect_changes(face_syndrome), d.detect_changes(vert_syndrome)

        # Dubugging matching
        try:
            matches = d.match_errors(matching_graph, [np.array(face_detections), np.array(vert_detections)])
        except Exception as e:
            print('Face\n', face_detections, '\n\nVertex \n' ,vert_detections, '\n\n')
            raise e

        predicted_errors = so.extract_predicted_errors(matches, self.dist, rounds)
        error = self.check_error(state_v.ex, state_v.ez, state_h.ex, state_h.ez, predicted_errors)

        return error, state_v, state_h, predicted_errors

    def run_round(
            self,
            state_v,
            state_h,
            op_dim_v,
            op_dim_h,
            noise_v,
            noise_h,
            face_meas_noise,
            vert_meas_noise,
            px,
            py,
            pz,
            pm,
        ):
         # Apply for X or Y
        x_noise_v = np.argwhere(noise_v < px + py)
        x_noise_h = np.argwhere(noise_h < px + py)

        # Apply for Y or Z
        z_noise_v = np.argwhere((noise_v < px + py + pz) & (noise_v >= px))
        z_noise_h = np.argwhere((noise_h < px + py + pz) & (noise_h >= px))

        oXv, oZv = o.X(op_dim_v, x_noise_v), o.Z(op_dim_v, z_noise_v)
        oXh, oZh= o.X(op_dim_h, x_noise_h), o.Z(op_dim_h, z_noise_h)
        
        state_v = state_v * oXv * oZv
        state_h = state_h * oXh * oZh

        face_syndrome_t, vert_syndrome_t = self.get_syndromes(state_v.ex, state_v.ez, state_h.ex, state_h.ez)

        face_syndrome_t = face_syndrome_t.astype(int) ^ (face_meas_noise < pm)
        vert_syndrome_t = vert_syndrome_t.astype(int) ^ (vert_meas_noise < pm)

        return face_syndrome_t, vert_syndrome_t, state_v, state_h

    def generate_noise(self, rounds):
        noise_full_v = self.rng.random((rounds, self.dist-1, self.dist-1))
        noise_full_h = self.rng.random((rounds, self.dist, self.dist))

        face_meas_noise_full = self.rng.random((rounds, self.dist-1, self.dist))
        vert_meas_noise_full = self.rng.random((rounds, self.dist, self.dist-1))

        return noise_full_v, noise_full_h, face_meas_noise_full, vert_meas_noise_full


    def get_syndromes(self, exv, ezv, exh, ezh):
        syn_face = exh[:-1] ^ exh[1:]
        syn_face[:, 1:]  ^= exv
        syn_face[:, :-1] ^= exv

        syn_vert = ezh[:, :-1] ^ ezh[:, 1:]
        syn_vert[1:]  ^= ezv
        syn_vert[:-1] ^= ezv
        return syn_face, syn_vert
    

    def check_error(
            self,
            true_x_errors_v,
            true_z_errors_v,
            true_x_errors_h,
            true_z_errors_h,
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

        x_errors_v, x_errors_h = predicted_errors[0], predicted_errors[1]
        combined_x_v, combined_x_h = np.array(x_errors_v ^ true_x_errors_v), np.array(x_errors_h ^ true_x_errors_h)

        z_errors_v, z_errors_h = predicted_errors[2], predicted_errors[3]
        combined_z_v, combined_z_h = np.array(z_errors_v ^ true_z_errors_v), np.array(z_errors_h ^ true_z_errors_h)

        errors = [0, 0]

        # Checking one col for Z error
        if sum(combined_z_h[:, 0]) % 2 == 1:
            errors[0] = 1 
            is_error = 1

        # Checking one row for X error
        if sum(combined_x_h[0, :]) % 2 == 1:
            errors[1] = 1 
            is_error = 1

        return is_error, errors
