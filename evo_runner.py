#!/usr/bin/env python3
# evo_runner.py

from __future__ import annotations
import json

# Old-style config types you already have
from ga.ga_types import GARunConfig, MutateParams, CrossoverParams

# New runner + deps
from ga.ga_runner import GARunner, RunnerDeps
from ga.ga_controls import ControlPool
from ga.ga_evaluator import PopulationEvaluator
from ga.ga_mutation import Mutator
from ga.ga_crossover import CrossoverOperator

# ANSI colors
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "END": "\033[0m",
}

def color_print(text: str, color: str = "GREEN", bold: bool = False) -> None:
    style = COLORS["BOLD"] if bold else ""
    print(f"{style}{COLORS[color]}{text}{COLORS['END']}")

# ---------- Compat shim so GARunner sees the fields it expects ----------
class CompatCfg:
    def __init__(self, old: GARunConfig, verbose_game: int = 0):
        self.population_size = getattr(old, "pop_size")
        self.generations = getattr(old, "generations")
        self.games_per_eval = getattr(old, "games_per_eval")
        self.elitism = getattr(old, "elitism")
        self.tournament_size = getattr(old, "tourney_size")
        self.seed = getattr(old, "eval_seed", None)
        self.verbose_game = verbose_game

        # If your GARunner ever reads these, they’re here; otherwise harmless.
        mp = getattr(old, "mutate_params", None)
        cp = getattr(old, "crossover_params", None)
        self.mutation_sigma_frac = getattr(mp, "sigma_frac", 0.1) if mp else 0.1
        self.mutation_prob = getattr(mp, "per_gene_prob", 0.2) if mp else 0.2
        self.reset_prob = getattr(mp, "reset_prob", 0.0) if mp else 0.0
        self.crossover_alpha_min = getattr(cp, "alpha_min", 0.3) if cp else 0.3
        self.crossover_alpha_max = getattr(cp, "alpha_max", 0.7) if cp else 0.7
        self.crossover_prob = getattr(cp, "per_gene_prob", 0.9) if cp else 0.9

# ---------- Robust builders that try multiple APIs ----------
def build_mutator(mp: MutateParams) -> Mutator:
    # 1) Known factory names
    for fname in ("from_params", "from_config", "make", "create", "build"):
        if hasattr(Mutator, fname):
            try:
                return getattr(Mutator, fname)(mp)
            except TypeError:
                pass

    # 2) Try passing the config object directly
    try:
        return Mutator(mp)
    except TypeError:
        pass

    # 3) Keyword variants
    kw_variants = [
        dict(sigma_frac=getattr(mp, "sigma_frac", None),
             per_gene_prob=getattr(mp, "per_gene_prob", None),
             reset_prob=getattr(mp, "reset_prob", None)),
        dict(sigma=getattr(mp, "sigma_frac", None),
             per_gene=getattr(mp, "per_gene_prob", None),
             reset=getattr(mp, "reset_prob", None)),
        dict(sigma_fraction=getattr(mp, "sigma_frac", None),
             per_gene_probability=getattr(mp, "per_gene_prob", None),
             reset_probability=getattr(mp, "reset_prob", None)),
    ]
    for kwargs in kw_variants:
        # remove Nones so we don't pass unknown names unnecessarily
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            if kwargs:
                return Mutator(**kwargs)
        except TypeError:
            continue

    # 4) Positional fallbacks
    pos = (getattr(mp, "sigma_frac", 0.1),
           getattr(mp, "per_gene_prob", 0.2),
           getattr(mp, "reset_prob", 0.0))
    for args in (pos, pos[:2], tuple()):
        try:
            return Mutator(*args)
        except TypeError:
            continue

    # 5) Absolute fallback: default-construct, then setattr if attributes exist
    m = Mutator()
    for k in ("sigma_frac", "per_gene_prob", "reset_prob"):
        if hasattr(m, k) and hasattr(mp, k):
            setattr(m, k, getattr(mp, k))
    return m

def build_crosser(cp: CrossoverParams) -> CrossoverOperator:
    # 1) Factories
    for fname in ("from_params", "from_config", "make", "create", "build"):
        if hasattr(CrossoverOperator, fname):
            try:
                return getattr(CrossoverOperator, fname)(cp)
            except TypeError:
                pass

    # 2) Try passing config directly
    try:
        return CrossoverOperator(cp)
    except TypeError:
        pass

    # 3) Keyword variants
    kw_variants = [
        dict(alpha_min=getattr(cp, "alpha_min", None),
             alpha_max=getattr(cp, "alpha_max", None),
             per_gene_prob=getattr(cp, "per_gene_prob", None)),
        dict(alpha_lo=getattr(cp, "alpha_min", None),
             alpha_hi=getattr(cp, "alpha_max", None),
             per_gene=getattr(cp, "per_gene_prob", None)),
    ]
    for kwargs in kw_variants:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            if kwargs:
                return CrossoverOperator(**kwargs)
        except TypeError:
            continue

    # 4) Positional fallbacks
    pos = (getattr(cp, "alpha_min", 0.3),
           getattr(cp, "alpha_max", 0.7),
           getattr(cp, "per_gene_prob", 0.9))
    for args in (pos, pos[:2], tuple()):
        try:
            return CrossoverOperator(*args)
        except TypeError:
            continue

    return CrossoverOperator()

# ---------- Main ----------
if __name__ == "__main__":
    color_print("\n=== Genetic Algorithm Configuration ===", "HEADER", bold=True)

    base_cfg = GARunConfig(
        pop_size=35,
        generations=200,
        games_per_eval=1500,
        elitism=10,
        tourney_size=8,
        mutation_after_crossover=True,
        mutate_params=MutateParams(sigma_frac=0.12, per_gene_prob=0.35, reset_prob=0.06),
        crossover_params=CrossoverParams(alpha_min=0.3, alpha_max=0.7, per_gene_prob=0.9),
        eval_seed=42,
        log_every=1,
    )

    print(
        f"""
    Population Size: {COLORS['CYAN']}{base_cfg.pop_size}{COLORS['END']}
    Generations: {COLORS['CYAN']}{base_cfg.generations}{COLORS['END']}
    Games per Evaluation: {COLORS['YELLOW']}{base_cfg.games_per_eval}{COLORS['END']}
    Elitism: {COLORS['GREEN']}{base_cfg.elitism}{COLORS['END']}
    Tournament Size: {COLORS['GREEN']}{base_cfg.tourney_size}{COLORS['END']}
    Mutation:
      Sigma Fraction: {COLORS['RED']}{base_cfg.mutate_params.sigma_frac}{COLORS['END']}
      Per-Gene Probability: {COLORS['RED']}{base_cfg.mutate_params.per_gene_prob}{COLORS['END']}
      Reset Probability: {COLORS['RED']}{base_cfg.mutate_params.reset_prob}{COLORS['END']}
    Crossover:
      Alpha Range: {COLORS['BLUE']}[{base_cfg.crossover_params.alpha_min}, {base_cfg.crossover_params.alpha_max}]{COLORS['END']}
      Per-Gene Probability: {COLORS['BLUE']}{base_cfg.crossover_params.per_gene_prob}{COLORS['END']}
        """.rstrip()
    )

    # Build deps (keep only MetaOne if your ControlPool supports it)
    controls = ControlPool()  # or ControlPool(only=["MetaOne"])
    evaluator = PopulationEvaluator(controls)
    mutator = build_mutator(base_cfg.mutate_params)
    crosser = build_crosser(base_cfg.crossover_params)

    deps = RunnerDeps(
        evaluator=evaluator,
        mutator=mutator,
        crosser=crosser,
        controls=controls,
    )

    # Shim config so GARunner sees expected attribute names
    cfg = CompatCfg(base_cfg, verbose_game=0)

    color_print("=== Starting Evolution ===", "HEADER", bold=True)
    runner = GARunner(cfg, deps)
    best_g, best_f = runner.evolve()

    color_print("\n=== Best Genome ===", "HEADER", bold=True)
    payload = best_g.to_json() if hasattr(best_g, "to_json") else getattr(best_g, "data", {})
    print(json.dumps(payload, indent=2))

    color_print("\n=== Best Fitness ===", "HEADER", bold=True)
    bust_rate = 0.0
    if hasattr(best_f, "diagnostics") and isinstance(best_f.diagnostics, dict):
        bust_rate = float(best_f.diagnostics.get("bust_rate", 0.0))
    print(
        f"{COLORS['BOLD']}Median Score:{COLORS['END']} {COLORS['GREEN']}{getattr(best_f, 'median', 0.0):.2f}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Win Rate:{COLORS['END']} {COLORS['CYAN']}{100*getattr(best_f, 'win_rate', 0.0):.2f}%{COLORS['END']}\n"
        f"{COLORS['BOLD']}Max Score:{COLORS['END']} {COLORS['YELLOW']}{getattr(best_f, 'max_score', 0)}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Min Score:{COLORS['END']} {COLORS['RED']}{getattr(best_f, 'min_score', 0)}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Bust Rate:{COLORS['END']} {COLORS['RED']}{100*bust_rate:.2f}%{COLORS['END']}"
    )
