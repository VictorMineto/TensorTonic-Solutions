import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    if not x:
        return None, None, None

    mean_val = sum(x) / len(x)
    

    sorted_x = sorted(x)
    n = len(sorted_x)
    if n % 2 == 0:
        median_val = (sorted_x[n//2 - 1] + sorted_x[n//2]) / 2
    else:
        median_val = sorted_x[n//2]
    

    counts = Counter(sorted_x)
    max_count = max(counts.values())
    modes = [value for value, count in counts.items() if count == max_count]
    mode_val = min(modes)
    
    return mean_val, median_val, mode_val