# ga/pod_population_evaluator.py
from __future__ import annotations
from typing import List, Tuple, Dict
from .ga_types import Fitness
from .ga_genome import Genome
from .pod_evaluator import PodEloEvaluator

class PodPopulationEvaluator:
    """
    Adapter so the pod-based evaluator matches the GA runner's expected API:
      evaluate_all_in_one(pop, games_per_eval, base_seed, verbose_game)
    We map games_per_eval -> pod 'rounds'.
    """
    def __init__(self, pod_size: int = 5, k_factor: float = 24.0):
        self.pod_size = pod_size
        self.k_factor = k_factor
        self._impl = PodEloEvaluator(pod_size=pod_size, k_factor=k_factor)

    def evaluate_all_in_one(
        self,
        pop: List[Genome],
        games_per_eval: int = 100,   # interpreted as "rounds"
        base_seed: int = 123,
        verbose_game: int = 0,
    ) -> Tuple[List[Fitness], Dict[str, Fitness]]:
        rounds = max(1, int(games_per_eval))
        fits, control = self._impl.evaluate(
            pop=pop,
            rounds=rounds,
            base_seed=base_seed,
            verbose_game=verbose_game,
        )
        return fits, control
