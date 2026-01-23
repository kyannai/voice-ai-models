"""
Common utility functions for ASR benchmarks.
"""

import random
from typing import Any, Dict, List


def split_dict(data: Dict[str, Any], num_splits: int) -> List[Dict[str, Any]]:
    """
    Split a dictionary into N roughly equal parts.
    
    Keys are shuffled before splitting to ensure balanced workloads.
    
    Args:
        data: Dictionary to split
        num_splits: Number of parts to split into
        
    Returns:
        List of dictionaries, each containing a subset of the original
    """
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]
