#!/usr/bin/env python3
# evo_pods_runner.py
from __future__ import annotations
import json

from ga.ga_types import GARunConfig, MutateParams, CrossoverParams
from ga.ga_runner import GARunner, RunnerDeps
from ga.ga_controls import ControlPool
from ga.ga_mutation import Mutator
from ga.ga_crossover import CrossoverOperator
from ga.pod_population_evaluator import PodPopulationEvaluator

# ANSI colors for a nice header (optional)
COLORS = {
    "HEADER": "\033[95m", "BLUE": "\033[94m", "CYAN": "\033[96m",
    "GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m",
    "BOLD": "\033[1m", "UNDERLINE": "\033[4m", "END": "\033[0m",
}
def cprint(text: str, color: str = "GREEN", bold: bool = False) -> None:
    style = COLORS["BOLD"] if bold else ""
    print(f"{style}{COLORS[color]}{text}{COLORS['END']}")

if __name__ == "__main__":
    # Tune these like you did for evo_runner; here games_per_eval == pod "rounds"
    cfg = GARunConfig(
        pop_size=35,
        generations=50,
        games_per_eval=120,   # <- rounds per generation in pod scheduling
        elitism=10,
        tourney_size=8,
        mutate_params=MutateParams(sigma_frac=0.12, per_gene_prob=0.35, reset_prob=0.06),
        crossover_params=CrossoverParams(alpha_min=0.3, alpha_max=0.7, per_gene_prob=0.9),
        eval_seed=42,
        log_every=1,
    )

    cprint("\n=== GA with Pod-Based Evaluation ===", "HEADER", bold=True)
    print(
        f"Population: {cfg.pop_size} | Generations: {cfg.generations} | "
        f"Rounds/Gen: {cfg.games_per_eval} | Elitism: {cfg.elitism} | "
        f"Tournament: {cfg.tourney_size}"
    )

    evaluator = PodPopulationEvaluator(pod_size=5, k_factor=24.0)
    mutator = Mutator(cfg.mutate_params)
    crosser = CrossoverOperator(cfg.crossover_params)
    controls = ControlPool()  # not used by pod evaluator, but fine for runner summary

    deps = RunnerDeps(
        evaluator=evaluator,
        mutator=mutator,
        crosser=crosser,
        controls=controls,
    )

    runner = GARunner(cfg, deps)
    best_g, best_f = runner.evolve()

    cprint("\n=== Best Genome (pods GA) ===", "HEADER", bold=True)
    payload = best_g.to_json() if hasattr(best_g, "to_json") else getattr(best_g, "data", {})
    print(json.dumps(payload, indent=2))

    cprint("\n=== Best Fitness ===", "HEADER", bold=True)
    bust_rate = float(best_f.diagnostics.get("bust_rate", 0.0)) if getattr(best_f, "diagnostics", None) else 0.0
    print(
        f"Median {best_f.median:.2f} | Win% {100*best_f.win_rate:.2f}% | "
        f"Q1 {best_f.q1:.2f} | Q3 {best_f.q3:.2f} | "
        f"Max {best_f.max_score} | Min {best_f.min_score} | "
        f"Bust% {100*bust_rate:.2f}%"
    )
