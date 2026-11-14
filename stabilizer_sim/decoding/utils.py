from typing import Union, List

import numpy as np
from pymatching import Matching

def detect_changes(syndrome: list) -> list:
    syndrome = np.array(syndrome)

    prepend = np.zeros((1,) + syndrome.shape[1:], dtype=syndrome.dtype)
    syndrome_change = np.abs(np.diff(syndrome, axis=0, prepend=prepend))

    return syndrome_change


def match_errors(matching_graph: Matching, detection_events) -> list:
    #TODO: makwe sure working for toric
    matched = matching_graph.decode_to_matched_dets_array(np.concat([e.flatten() for e in detection_events]))

    return matched

