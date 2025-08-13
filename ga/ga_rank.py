# ga/ga_rank.py

from typing import List

def rank_desc(values: List[float]) -> List[float]:
    """
    Return 1-based ranks for `values` with average ranks for ties.
    Larger value = better rank (1 = best).
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j+1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks
