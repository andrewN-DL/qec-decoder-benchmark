from typing import Union, List

import numpy as np
from pymatching import Matching


def build_matching_graph(distance, rounds, px, py, pz, pm):
    """
    For now assuming a square graph
    """
    matching = Matching()

    # To add final noiseless layer -> classical extraction
    rounds += 1

    # Face + Vertex
    num_face_stabilizers = distance * (distance - 1)
    num_vert_stabilizers = (distance - 1) * distance

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

    h_dist_face = distance
    v_dist_face = distance - 1

    h_dist_vert = distance -1
    v_dist_vert = distance

    for r in range(rounds):
        # Face operators
        for vd in range(v_dist_face):
            for hd in range(h_dist_face):
                # Adding horizontal face edges

                if hd < h_dist_face - 1:
                    org = r*num_face_stabilizers + vd*h_dist_face + hd
                    dest = org + 1

                    matching.add_edge(org, dest, weight=weightx)
                if vd < v_dist_face - 1:
                    org = r*num_face_stabilizers + vd*h_dist_face + hd
                    dest = org + h_dist_face

                    matching.add_edge(org, dest, weight=weightx)
                if vd in [0, v_dist_face - 1]:

                    matching.add_boundary_edge(r*num_face_stabilizers + vd*h_dist_face + hd, weight=weightx)

        for vd in range(v_dist_vert):
            for hd in range(h_dist_vert):

                # Adding horizontal face edges
                if hd < h_dist_vert - 1:
                    org = r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face
                    dest = org + 1

                    matching.add_edge(org, dest, weight=weightz)
                if vd < v_dist_vert - 1:
                    org = r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face
                    dest = org + h_dist_vert

                    matching.add_edge(org, dest, weight=weightz)
                if hd in [0, h_dist_vert - 1]:
                    matching.add_boundary_edge(r*num_vert_stabilizers + vd*h_dist_vert + hd + num_detectors_face, weight=weightz)
            
            
            # Adding measurement edges (currently no time boundry TODO)
        if r < rounds - 1:
            for s in range(num_face_stabilizers):
                matching.add_edge(
                    (num_face_stabilizers*r) + s,
                    (num_face_stabilizers*(r+1)) + s,
                    weight=weightm
                )
            for s in range(num_vert_stabilizers):

                matching.add_edge(
                    (num_vert_stabilizers*r) + s + num_detectors_face,
                    (num_vert_stabilizers*(r+1)) + s + num_detectors_face,
                    weight=weightm
                )

    return matching


# NOTE: This currently assumes all data qubits have the same noise model -> Shortest path = minimum weight
# NOTE: Also assumes all errors uncorrelated
def extract_predicted_errors(matched_errors, distance, rounds):
    d = distance
    # Face + Vertex
    num_face_stabilizers = d * (d - 1)
    num_vert_stabilizers = (d - 1) * d

    n_face_det = (rounds + 1) * num_face_stabilizers

    x_errors_h = np.zeros((d, d), dtype=np.uint8)
    x_errors_v = np.zeros((d-1, d-1), dtype=np.uint8)

    z_errors_h = np.zeros((d, d), dtype=np.uint8)
    z_errors_v = np.zeros((d-1, d-1), dtype=np.uint8)

    for u, v in matched_errors:
        u, v = int(u), int(v)

        # Match to the spatial boundary
        if u == -1 or v == -1:
            det = max(u, v)

            if det < n_face_det:
                # Face detector -> X errors, boundary on top/bottom
                i, j = divmod(det % num_face_stabilizers, d)
                if i + 1 <= d - 1 - i: 
                    x_errors_h[:i+1, j] ^= 1
                else:
                    x_errors_h[i+1:, j] ^= 1
            else:
                # Vertex detector -> Z errors, boundary on left/right
                i, j = divmod((det - n_face_det) % num_vert_stabilizers, d - 1)
                if j + 1 <= d - 1 - j:
                    z_errors_h[i, :j+1] ^= 1
                else:
                    z_errors_h[i, j+1:] ^= 1
            continue

        assert (u < n_face_det) == (v < n_face_det), 'X and Z errors matched'

        if u < n_face_det:
            # Face pair -> X errors
            i1, j1 = divmod(u % num_face_stabilizers, d)
            i2, j2 = divmod(v % num_face_stabilizers, d)
            if i1 > i2:
                (i1, j1), (i2, j2) = (i2, j2), (i1, j1)

            # Horizontal along row i1, then vertical down column j2
            x_errors_v[i1, min(j1, j2):max(j1, j2)] ^= 1
            x_errors_h[i1+1:i2+1, j2] ^= 1

        else:
            # Vertex pair -> Z errors
            i1, j1 = divmod((u - n_face_det) % num_vert_stabilizers, d - 1)
            i2, j2 = divmod((v - n_face_det) % num_vert_stabilizers, d - 1)
            if i1 > i2:
                (i1, j1), (i2, j2) = (i2, j2), (i1, j1)

            # Horizontal along row i1, then vertical down column j2
            z_errors_h[i1, min(j1, j2)+1:max(j1, j2)+1] ^= 1
            z_errors_v[i1:i2, j2] ^= 1


    return x_errors_v, x_errors_h, z_errors_v, z_errors_h