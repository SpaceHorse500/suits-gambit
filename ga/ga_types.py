# ga/ga_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

# ------------ Hyperparameter bundles ------------

@dataclass(frozen=True)
class MutateParams:
    """Mutation hyperparameters used by Mutator."""
    sigma_frac: float = 0.12        # stddev as fraction of each bound range
    per_gene_prob: float = 0.35         # probability a given gene is mutated
    reset_prob: float = 0.06            # chance to reset to random within bounds
    seed: int | None = None             # optional: RNG seed for reproducibility

@dataclass(frozen=True)
class CrossoverParams:
    """Crossover hyperparameters used by CrossoverOperator."""
    alpha_min: float = 0.30             # BLX/arith blend min alpha
    alpha_max: float = 0.70             # BLX/arith blend max alpha
    per_gene_prob: float = 0.90         # probability a given gene is crossed
    seed: int | None = None             # optional: RNG seed for reproducibility

# ------------ GA run configuration ------------

@dataclass
class GARunConfig:
    """Top-level configuration for a GA run."""
    pop_size: int = 30
    generations: int = 200
    games_per_eval: int = 500
    elitism: int = 10
    tourney_size: int = 8

    mutate_params: MutateParams = field(default_factory=MutateParams)
    crossover_params: CrossoverParams = field(default_factory=CrossoverParams)
    mutation_after_crossover: bool = True

    eval_seed: int = 1337
    log_every: int = 1

# ------------ Fitness container ------------

@dataclass
class Fitness:
    """
    Fitness used by the GA.
    - win_rate: primary objective.
    - median: secondary objective.
    - q1/q3: tertiary objectives to reward conservative/consistent performers.
    - diagnostics: extra info (sd, iqr, mean, etc.) for logging only.
    """
    win_rate: float
    median: float
    q1: float
    q3: float
    max_score: int
    min_score: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
