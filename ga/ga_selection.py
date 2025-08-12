# ga_selection.py
import copy, random
from typing import List, Tuple, Dict, Any, Optional
from .ga_types import Fitness

class FitnessComparator:
    def __init__(self, median_tol: float = 0.2, win_tol: float = 0.005):
        self.median_tol = median_tol
        self.win_tol = win_tol

    def better(self, a: Fitness, b: Fitness) -> bool:
        if abs(a.median - b.median) > self.median_tol:
            return a.median > b.median
        if abs(a.win_rate - b.win_rate) > self.win_tol:
            return a.win_rate > b.win_rate
        if a.max_score != b.max_score:
            return a.max_score > b.max_score
        if a.min_score != b.min_score:
            return a.min_score > b.min_score
        return a.diagnostics.get("bust_rate", 1.0) < b.diagnostics.get("bust_rate", 1.0)

class Selector:
    def __init__(self, comparator: FitnessComparator):
        self.comparator = comparator

    def tournament(self, pop: List[Tuple[Dict[str, Any], Fitness]],
                   tourney_size: int = 4,
                   k: int = 1,
                   rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
        rng = rng or random
        selected: List[Dict[str, Any]] = []
        for _ in range(k):
            group = rng.sample(pop, tourney_size)
            best_g, best_f = group[0]
            for g, f in group[1:]:
                if self.comparator.better(f, best_f):
                    best_g, best_f = g, f
            selected.append(copy.deepcopy(best_g))
        return selected
