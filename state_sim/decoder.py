import numpy as np
from repitition_code import RepititionCode
from circuit import Circuit

from pymatching import Matching
import networkx as nx

def detect_changes(syndrome: list, possible_errors: list=[]) -> list:
    initial_error = np.array([])
    syndrome = np.array(syndrome)

    syndrome_change = np.abs(np.diff(syndrome, axis=0, prepend=np.zeros((1, syndrome.shape[1]))))

    return syndrome_change

def decode(detection_events: list, possible_errors: list=[]) -> list:
    '''
    Mechanism:

    We list all possible weight 1 errors and their associated detector patterns
    waituntil t=3 to start decoding
    '''
    detection_events = np.array(detection_events)

    possible_erorrs = {
        'X1': [[1, 0]],
        'X2': [[1, 1]],
        'X3': [[0, 1]],
        'M1': [[1, 0], [1, 0]],
        'M1': [[0, 1], [0, 1]],
    }

    error_combinations = np.array([np.array(a[0]) ^ np.array(a[1]) for a in zip(possible_erorrs['M1'], possible_erorrs['M2'])])

    current_window = []
    current_error = 'I'
    total_errors = []
    for i, event in enumerate(detection_events):
        if current_window not in possible_errors.values():
            if [event] not in possible_errors.values():
                current_window.append(event)
            elif [event] == [[0, 0]]:
                current_window = []
                total_errors.append(current_error)
            else:
                for k, v in possible_erorrs.items():
                    if v == [event]:    
                        total_errors.append(k)


    return []
def match_decode(detection_events: list, possible_errors: list=[]) -> list:
    '''
    Instead of looking at looking at the last n time steps, we can look at whole space time 'grid'
    Put detections of the vertices of the grid
    grid also has legs on the edges connecting to nothing
    physical errors give either one or two detection events. These events only last for one timestep (qubits should remain in same state if no error)
    measurement errors give two detection events, spread accross two timesteps (ancillas are reset when errors occur)
    if two errors happen on the same stabilizer (measurement error and one of its physical qubits flips, or both physical qubits flip ) it cancels out in the syndrome
    detection errors only show if they have an odd number of errors associated to them
    assume each detection error (vertices) only has one error associated with it (mathematically more likely)
    find a matching between detection errors (following grid lines) 
    this includes between points along an edge which connects nothing
    prob of a given edge is given by px or pm (probs of physical or matching errors)
    if a match goes accross more than one edge, we multiple probabilities of both edges to get total prob
    want to find matching with highest probability

    IDEAS
    if zero for more than k turns, 'cut' the graph. Assume no cumulative measurement errors

    Option 1: MILP
    Option 2: Blossom algorithm
        - Change/wrap Nx
        - Pymatching  
    Option 3: Genetic algorithm
    Option 4: RL/ML
    Option 5: Brute force
    '''
    # Physical and measurement probabilities
    px = 0.1
    pm = 0.05


    points = []
    for i, event in enumerate(detection_events):
        if 1 in event:
            for j, meas in enumerate(event):
                if meas == 1:
                    points.append([i, j])

    errors = {i: err for i, err in enumerate(points)}
    print(errors)

    edges = [(a, b) for a in errors.keys() for b in errors.keys()]

    weights = {}

    # Building weighted graph
    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]

        # Prevent double counting
        if v2 < v1:
            continue
        e1, e2 = errors[v1], errors[v2]

        # Represents matching with boundary
        if v1 == v2:
            # Flip as we want to work with positive numbers
            weights[edge] = np.log((1 - px)/px)
            continue

        diff = np.array(e2) - np.array(e1)

        # Portion contribution per edge to error of given configuration (assume manhattan distance - most likely)
        # True prob is a constant * exp(-sum(selected weights))
        weights[edge] = abs(diff[0]) * np.log((1 - pm)/(pm)) + abs(diff[1]) * np.log((1 - px) / px)

    G = nx.Graph()

    for i, (edge, weight) in enumerate(weights.items()):
        if edge[0] != edge[1]:
            G.add_edge(edge[0], edge[1], weight=weight)
        else:
            G.add_edge(edge[0], edge[0]+9, weight=weight)

    print(G.edges)

    matched = nx.min_weight_matching(G)

    return matched


def match_decode(detection_events: list, possible_errors: list=[]) -> list:
    '''
    Instead of looking at looking at the last n time steps, we can look at whole space time 'grid'
    Put detections of the vertices of the grid
    grid also has legs on the edges connecting to nothing
    physical errors give either one or two detection events. These events only last for one timestep (qubits should remain in same state if no error)
    measurement errors give two detection events, spread accross two timesteps (ancillas are reset when errors occur)
    if two errors happen on the same stabilizer (measurement error and one of its physical qubits flips, or both physical qubits flip ) it cancels out in the syndrome
    detection errors only show if they have an odd number of errors associated to them
    assume each detection error (vertices) only has one error associated with it (mathematically more likely)
    find a matching between detection errors (following grid lines) 
    this includes between points along an edge which connects nothing
    prob of a given edge is given by px or pm (probs of physical or matching errors)
    if a match goes accross more than one edge, we multiple probabilities of both edges to get total prob
    want to find matching with highest probability

    IDEAS
    if zero for more than k turns, 'cut' the graph. Assume no cumulative measurement errors

    Option 1: MILP
    Option 2: Blossom algorithm
        - Change/wrap Nx
        - Pymatching  
    Option 3: Genetic algorithm
    Option 4: RL/ML
    Option 5: Brute force
    '''
    # Physical and measurement probabilities
    px = 0.1
    pm = 0.05

    matching = Matching()

    for i in range(len(detection_events)):
        # Adding physical edges
        matching.add_boundary_edge(2*i, weight=px)
        matching.add_edge(2*i, 2*i+1, weight=px)
        matching.add_boundary_edge(2*i+1, weight=px)

        # Adding measurement edges (currently no time boundry TODO)
        if i < len(detection_events) - 1:
            matching.add_edge(2*i, 2*(i + 1), weight=pm)
            matching.add_edge(2*i + 1, 2*(i + 1) + 1, weight=pm)

    matched = matching.decode_to_matched_dets_array(detection_events.flatten())
    
    return matched


# code = RepititionCode(3, seed=42)
# syndrome = code.run(
#     error_prob=0.04,
#     error_weight=3,
#     measurement_noise_prob=0.04,
#     shots=1,
#     rounds=50,
#     random_initial_state=True
# )

# detections = detect_changes(syndrome)
# code.display_syndrome(detections)
# print(match_decode(detections))
