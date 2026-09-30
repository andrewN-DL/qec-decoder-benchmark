from typing import Union, List

import numpy as np
from pymatching import Matching

def detect_changes(syndrome: list) -> list:
    syndrome = np.array(syndrome)

    syndrome_change = np.abs(np.diff(syndrome, axis=0, prepend=np.zeros((1, syndrome.shape[1]))))

    return syndrome_change


def match_errors(matching_graph: Matching, detection_events) -> list:

    matched = matching_graph.decode_to_matched_dets_array(detection_events.flatten())

    return matched

