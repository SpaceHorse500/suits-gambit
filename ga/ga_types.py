# ga_types.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class MutateParams:
    sigma_frac: float = 0.10
    per_gene_prob: float = 0.30
    reset_prob: float = 0.05
    seed: Optional[int] = None

@dataclass
class CrossoverParams:
    alpha_min: float = 0.25
    alpha_max: float = 0.75
    per_gene_prob: float = 0.90
    seed: Optional[int] = None

@dataclass
class Fitness:
    median: float
    win_rate: float
    max_score: int
    min_score: int
    diagnostics: Dict[str, Any]

@dataclass
class GARunConfig:
    pop_size: int = 50
    generations: int = 20
    games_per_eval: int = 500           # shared games per generation
    elitism: int = 5
    tourney_size: int = 4
    mutation_after_crossover: bool = True
    mutate_params: MutateParams = field(default_factory=MutateParams)
    crossover_params: CrossoverParams = field(default_factory=CrossoverParams)
    eval_seed: int = 123
    log_every: int = 1
