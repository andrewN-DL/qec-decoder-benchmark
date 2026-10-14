from typing import Union, List

import numpy as np
from pymatching import Matching


def build_toric_matching_graph(distance, rounds, px, pz, pm):
    """
    For now assuming a square graph
    """
    matching = Matching()

    # Not omitting final 2 stabilizers for now as easier (although not independent)
    num_z_stabilizers = distance**2

    # d**2 spatial and temporal for all rounds. No temporal on last
    num_detectors_z = (rounds) * (distance**2)

    if px == 0:
        px = 1e-6
    if pz == 0:
        pz = 1e-6
    if pm == 0:
        pm = 1e-6
    
    weightx = -np.log(px/(1-px))
    weightz = -np.log(pz/(1-pz))
    weightm = -np.log(pm/(1-pm))

    for r in range(rounds):
        for i in range(distance):
            for j in range(distance):
                # Horizontal connection (Z stabilizers)
                org = (i*distance + j) + r*(num_z_stabilizers)
                dest = (i*distance + j) + r*(num_z_stabilizers) + 1
                if dest % distance == 0:
                    dest -= distance
                matching.add_edge(org, dest, weight=weightx)

                # Vertical connection (Z stabilizers)
                org = (i*distance + j) + r*(num_z_stabilizers)
                dest = (i + 1)*distance + j + r*(num_z_stabilizers)
                if dest % num_z_stabilizers < distance:
                    dest -= num_z_stabilizers
                matching.add_edge(org, dest, weight=weightx)

                # Horizontal connection (X stabilizers)
                org = (i*distance + j) + r*(num_z_stabilizers) + num_detectors_z
                dest = (i*distance + j) + r*(num_z_stabilizers) + 1 + num_detectors_z
                if dest % distance == 0:
                    dest -= distance
                matching.add_edge(org, dest, weight=weightz)

                # Vertical connection (X stabilizers)
                org = (i*distance + j) + r*(num_z_stabilizers) + num_detectors_z
                dest = ((i + 1)*distance + j) + r*(num_z_stabilizers) + num_detectors_z
                if dest % num_z_stabilizers < distance:
                    dest -= num_z_stabilizers
                matching.add_edge(org, dest, weight=weightz)

        # Adding measurement edges (currently no time boundry TODO)
        if r < rounds - 1:
            for s in range(distance**2):
                matching.add_edge(
                    (num_z_stabilizers*r)+s,
                    (num_z_stabilizers*(r+1))+s,
                    weight=weightm)
                
                matching.add_edge(
                    (num_z_stabilizers*r)+s+num_detectors_z,
                    (num_z_stabilizers*(r+1))+s+num_detectors_z,
                    weight=weightm
                )

    return matching


# def extract_toric_predicted_errors(matched_errors, distance, rounds):
#     num_z_stabilizers = distance ** 2
#     num_z_detectors = rounds * num_z_stabilizers

#     x_errors = np.zeros([2*distance, distance])
#     z_errors = np.zeros([2*distance, distance])

#     for match in matched_errors:

#         assert (match[0] < num_z_detectors) == (match[1] < num_z_detectors), 'X and Z errors matched'

#         if match[0] < num_z_detectors:
#             error_type = 0
#         else:
#             error_type = 1
#             match = [match[0] - num_z_detectors, match[1] - num_z_detectors]

#         match = sorted(match)

#         s1, t1 = match[0] % num_z_stabilizers, match[0] // num_z_stabilizers
#         s2, t2 = match[1] % num_z_stabilizers, match[1] // num_z_stabilizers

#         # [43, 44] -> [7, 8] -> [2, 1], [2, 2]
#         # [44, 47] -> [8, 2] -> [2, 2], [0, 2]
#         i1, j1 = s1 // distance, s1 % distance
#         i2, j2 = s2 // distance, s2 % distance

#         print('Error Type', error_type)
#         print(i1, j1)
#         print(i2, j2, 'Two')

#         # v_x_errors = min(((i1-i2) % distance), ((i2-i1) % distance))
#         v_x_errors = i2 - i1
#         h_x_errors = j2 - j1

#         print('dist', v_x_errors, h_x_errors)

#         if ((i2-i1) // distance) < (distance / 2):
#             v_start = i1
#             v_end = i2
#         else:
#             v_start = i2
#             v_end = i1

#         for er in range(v_x_errors):
#             if error_type == 0: 
#                 x_errors[2*(v_start + er) % (2 * distance), j1] += 1
#             else:
#                 z_errors[((2*(v_start) + er) + 1) % (distance * 2), j2] += 1

#         if ((j2-j1) % distance) < (distance / 2): 
#             h_start = j1
#         else:
#             h_start = j2

#         for er in range(h_x_errors):
#             if error_type == 0:
#                 x_errors[2*(v_end) + 1, (h_start + (er)) % distance] += 1
#             else:
#                 z_errors[2*((v_end)), (h_start + er) % distance] += 1

#     return x_errors % 2, z_errors % 2


def extract_toric_predicted_errors(matched_errors, distance, rounds):
    d = distance
    n_stab = d * d
    n_z_det = rounds * n_stab

    x_errors = np.zeros((2*d, d), dtype=np.uint8)
    z_errors = np.zeros((2*d, d), dtype=np.uint8)

    def signed_step(a, b):
        # shortest signed displacement a -> b on a cycle of length d
        return ((b - a + d // 2) % d) - d // 2

    for u, v in matched_errors:
        u, v = int(u), int(v)
        assert (u < n_z_det) == (v < n_z_det), 'X and Z errors matched'

        is_x = u < n_z_det
        if not is_x:
            u -= n_z_det
            v -= n_z_det

        s1, s2 = u % n_stab, v % n_stab
        i1, j1 = divmod(s1, d)
        i2, j2 = divmod(s2, d)

        # vertical leg: column j1, from row i1 to row i2
        dv = signed_step(i1, i2)
        step = 1 if dv > 0 else -1
        i = i1
        for _ in range(abs(dv)):
            k = i if step > 0 else (i - 1) % d
            if is_x:
                x_errors[2*k + 1, j1] ^= 1
            else:
                z_errors[2*((k + 1) % d), j1] ^= 1
            i = (i + step) % d

        # horizontal leg: row i2, from column j1 to column j2
        dh = signed_step(j1, j2)
        step = 1 if dh > 0 else -1
        j = j1
        for _ in range(abs(dh)):
            k = j if step > 0 else (j - 1) % d
            if is_x:
                x_errors[2*i2, k] ^= 1
            else:
                z_errors[2*i2 + 1, (k + 1) % d] ^= 1
            j = (j + step) % d

    return x_errors, z_errors