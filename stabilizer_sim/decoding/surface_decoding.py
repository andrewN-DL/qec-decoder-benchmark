from typing import Union, List

import numpy as np
from pymatching import Matching


def build_toric_matching_graph(distance, rounds, px, py, pz, pm):
    """
    For now assuming a square graph
    """
    matching = Matching()

    # To add final noiseless layer -> classical extraction
    rounds += 1

    # Face + Vertex
    num_face_stabilizers = (distance * (distance + 2))
    num_vert_stabilizers = (distance + 1)**2

    # d**2 spatial and temporal for all rounds. No temporal on last
    num_detectors_face = (rounds) * num_face_stabilizers

    if px == 0:
        px = 1e-9
    if pz == 0:
        pz = 1e-9
    if pm == 0:
        pm = 1e-9

    px = px + py
    pz = pz + py
    
    weightx = -np.log(px/(1-px))
    weightz = -np.log(pz/(1-pz))
    weightm = -np.log(pm/(1-pm))

    # Face operator boundaries along smooth edges
    # Vertex operator boundaries along rough edges

    h_dist_face = distance + 2
    v_dist_face = distance

    h_dist_vert = distance + 1
    v_dist_vert = distance + 1

    print(rounds)

    for r in range(rounds):
        # Face operators
        for vd in range(v_dist_face):
            for hd in range(h_dist_face):
                # Adding horizontal face edges
                print(r, vd, hd)
                if hd < h_dist_face - 1:
                    org = r*num_face_stabilizers + vd*h_dist_face + hd
                    dest = org + 1
                    # print('A', org, dest)
                    matching.add_edge(org, dest, weight=weightx)
                if vd < v_dist_face - 1:
                    org = r*num_face_stabilizers + vd*h_dist_face + hd
                    dest = org + h_dist_face
                    # print('B', org, dest)
                    matching.add_edge(org, dest, weight=weightx)
                if vd in [0, v_dist_face - 1]:
                    # print('BF', r*num_face_stabilizers + vd*h_dist_face + hd)
                    matching.add_boundary_edge(r*num_face_stabilizers + vd*h_dist_face + hd)
        # print('Enter Vert')
        for vd in range(v_dist_vert):
            for hd in range(h_dist_vert):
                print(r, vd, hd)
                # Adding horizontal face edges
                if hd < h_dist_vert - 1:
                    org = r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face
                    dest = org + 1
                    # print('C', org, dest)
                    matching.add_edge(org, dest, weight=weightx)
                if vd < v_dist_vert - 1:
                    org = r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face
                    dest = org + h_dist_vert
                    # print('D', org, dest)
                    matching.add_edge(org, dest, weight=weightx)
                if hd in [0, h_dist_vert - 1]:

                    # print('BV', r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face)
                    matching.add_boundary_edge(r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face)
            
            
            # Adding measurement edges (currently no time boundry TODO)
        if r < rounds - 1:
            for s in range(num_face_stabilizers):
                matching.add_edge(
                    (num_face_stabilizers*r) + s,
                    (num_face_stabilizers*(r+1)) + s,
                    weight=weightm
                )

                matching.add_edge(
                    (num_vert_stabilizers*r) + s + num_detectors_face,
                    (num_vert_stabilizers*(r+1)) + s + num_detectors_face,
                    weight=weightm
                )

    return matching




    # for r in range(rounds):
    #     for i in range(distance):
    #         for j in range(distance):
    #             # Horizontal connection (Z stabilizers)
    #             org = (i*distance + j) + r*(num_z_stabilizers)
    #             dest = (i*distance + j) + r*(num_z_stabilizers) + 1
    #             if dest % distance == 0:
    #                 dest -= distance
    #             matching.add_edge(org, dest, weight=weightx)

    #             # Vertical connection (Z stabilizers)
    #             org = (i*distance + j) + r*(num_z_stabilizers)
    #             dest = (i + 1)*distance + j + r*(num_z_stabilizers)
    #             if dest % num_z_stabilizers < distance:
    #                 dest -= num_z_stabilizers
    #             matching.add_edge(org, dest, weight=weightx)

    #             # Horizontal connection (X stabilizers)
    #             org = (i*distance + j) + r*(num_z_stabilizers) + num_detectors_z
    #             dest = (i*distance + j) + r*(num_z_stabilizers) + 1 + num_detectors_z
    #             if dest % distance == 0:
    #                 dest -= distance
    #             matching.add_edge(org, dest, weight=weightz)

    #             # Vertical connection (X stabilizers)
    #             org = (i*distance + j) + r*(num_z_stabilizers) + num_detectors_z
    #             dest = ((i + 1)*distance + j) + r*(num_z_stabilizers) + num_detectors_z
    #             if dest % num_z_stabilizers < distance:
    #                 dest -= num_z_stabilizers
    #             matching.add_edge(org, dest, weight=weightz)

    #     # Adding measurement edges (currently no time boundry TODO)
    #     if r < rounds - 1:
    #         for s in range(distance**2):
    #             matching.add_edge(
    #                 (num_z_stabilizers*r)+s,
    #                 (num_z_stabilizers*(r+1))+s,
    #                 weight=weightm)
                
    #             matching.add_edge(
    #                 (num_z_stabilizers*r)+s+num_detectors_z,
    #                 (num_z_stabilizers*(r+1))+s+num_detectors_z,
    #                 weight=weightm
    #             )



# NOTE: This currently assumes all data qubits have the same noise model -> Shortest path = minimum weight
def extract_toric_predicted_errors(matched_errors, distance, rounds):
    d = distance
    n_stab = d * d
    n_z_det = (rounds + 1) * n_stab

    x_errors = np.zeros((2*d, d), dtype=np.uint8)
    z_errors = np.zeros((2*d, d), dtype=np.uint8)

    def signed_step(a, b):
        # shortest signed displacement a -> b on a cycle of length d
        return ((b - a + d // 2) % d) - d // 2

    for u, v in matched_errors:

        # Continue if match with temporal boundary
        if u == -1 or v == -1:
            continue

        u, v = int(u), int(v)
        assert (u < n_z_det) == (v < n_z_det), 'X and Z errors matched'

        is_x = u < n_z_det
        if not is_x:
            u -= n_z_det
            v -= n_z_det

        s1, s2 = u % n_stab, v % n_stab

        # Continue if purely temporal
        if s1 == s2:
            continue

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