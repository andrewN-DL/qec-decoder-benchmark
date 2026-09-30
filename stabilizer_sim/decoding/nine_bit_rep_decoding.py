from typing import Union, List

import numpy as np
from pymatching import Matching


def build_9_qubit_matching_graph(rounds, px, pm):
    matching = Matching()
    num_stabilizers = 8
    num_graphs = 4
    stab_per_graph = int(num_stabilizers / num_graphs)

    if px > 0:
        weightx = -np.log(px/(1-px))
    if pm > 0:
        weightm = -np.log(pm/(1-pm))


    for i in range(rounds):
        # Adding physical edges
        if px > 0:
            for s in range(num_graphs):
                matching.add_boundary_edge((num_stabilizers*i)+(s*stab_per_graph), weight=weightx)
                matching.add_boundary_edge((num_stabilizers*i)+(s*stab_per_graph)+1, weight=weightx)
                matching.add_edge((num_stabilizers*i)+(s*stab_per_graph), (num_stabilizers*i)+(s*stab_per_graph)+1, weight=weightx)
                

        # Adding measurement edges (currently no time boundry TODO)
        if pm > 0:
            if i < rounds - 1:
                for s in range(num_stabilizers):
                    matching.add_edge((num_stabilizers*i)+s, (num_stabilizers*(i+1))+s, weight=weightm)

    return matching


def extract_9_qubit_predicted_errors(matched_errors):
    num_data_qubits = 9
    num_stabilizers = 8
    subgraphs = 3
    errors = np.zeros(num_data_qubits)
    for match in matched_errors:
        # TODO. This works specifically for 3 bit repitition code with bitflip noise
        s1, t1 = match[0] % num_stabilizers, match[0] // num_stabilizers
        s2, t2 = match[1] % num_stabilizers, match[1] // num_stabilizers

        subgraph = s1 // 2

        # TODO: need to deal with z-errors separately
        if -1 not in match:
            assert s1 // 2 == s2 //2

            if s1 - s2 != 0:
                m1, m2 = min(s1, s2), max(s1, s2)
                # for i in range(m1, m2):
                errors[(3*subgraph) + 1] += 1
        else:
            for i in match:
                if i != -1:
                    s = (i % num_stabilizers) % 2
                    
                    if s == 0:
                        errors[(3 * subgraph)] += 1
                    else:
                        errors[(3 * subgraph) + 2] += 1

    return errors % 2