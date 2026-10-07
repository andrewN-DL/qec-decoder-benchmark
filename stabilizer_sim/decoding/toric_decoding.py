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
    num_st_z = (rounds) * (distance**2)

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
                org = (i*distance + j) + r*(num_z_stabilizers) + num_st_z
                dest = (i*distance + j) + r*(num_z_stabilizers) + 1 + num_st_z
                if dest % distance == 0:
                    dest -= distance
                matching.add_edge(org, dest, weight=weightz)

                # Vertical connection (X stabilizers)
                org = (i*distance + j) + r*(num_z_stabilizers) + num_st_z
                dest = ((i + 1)*distance + j) + r*(num_z_stabilizers) + num_st_z
                if dest % num_z_stabilizers < distance:
                    dest -= num_z_stabilizers
                matching.add_edge(org, dest, weight=weightz)

        # Adding measurement edges (currently no time boundry TODO)
        if r < rounds - 1:
            for s in range(distance**2):
                matching.add_edge((num_z_stabilizers*r)+s, (num_z_stabilizers*(r+1))+s, weight=weightm)
                matching.add_edge((num_z_stabilizers*r)+s+num_st_z, (num_z_stabilizers*(r+1))+s+num_st_z, weight=weightm)

    return matching


def extract_toric_predicted_errors(matched_errors):
    num_data_qubits = 9
    num_stabilizers = 8
    subgraphs = 4
    x_errors = np.zeros(num_data_qubits)
    z_errors = np.zeros(int(num_data_qubits / 3))

    for match in matched_errors:
        s1, t1 = match[0] % num_stabilizers, match[0] // num_stabilizers
        s2, t2 = match[1] % num_stabilizers, match[1] // num_stabilizers

        subgraph = int(s1 // 2)

        if subgraph != 3:
            if -1 not in match:
                assert s1 // 2 == s2 //2

                if s1 - s2 != 0:
                    x_errors[(3*subgraph) + 1] += 1
            else:
                for i in match:
                    if i != -1:
                        s = int((i % num_stabilizers) % 2)
                        
                        if s == 0:
                            x_errors[(3 * subgraph)] += 1
                        else:
                            x_errors[(3 * subgraph) + 2] += 1
        else:
            if -1 not in match:
                assert s1 // 2 == s2 //2

                if s1 - s2 != 0:
                    z_errors[1] += 1
            else:
                for i in match:
                    if i != -1:
                        s = int((i % num_stabilizers) % 2)
                        
                        if s == 0:
                            z_errors[0] += 1
                        else:
                            z_errors[2] += 1


    return x_errors % 2, z_errors % 2