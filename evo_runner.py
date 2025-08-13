import json
from ga.ga_types import GARunConfig, MutateParams, CrossoverParams
from ga.ga_runner import GARunner

# ANSI color codes
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
    """Prints colored text with optional bold formatting."""
    style = COLORS["BOLD"] if bold else ""
    print(f"{style}{COLORS[color]}{text}{COLORS['END']}")

if __name__ == "__main__":
    # Config with colored output
    color_print("\n=== Genetic Algorithm Configuration ===", "HEADER", bold=True)
    cfg = GARunConfig(
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
    
    # Print config with colors
    config_str = f"""
    Population Size: {COLORS['CYAN']}{cfg.pop_size}{COLORS['END']}
    Generations: {COLORS['CYAN']}{cfg.generations}{COLORS['END']}
    Games per Evaluation: {COLORS['YELLOW']}{cfg.games_per_eval}{COLORS['END']}
    Elitism: {COLORS['GREEN']}{cfg.elitism}{COLORS['END']}
    Tournament Size: {COLORS['GREEN']}{cfg.tourney_size}{COLORS['END']}
    Mutation:
      Sigma Fraction: {COLORS['RED']}{cfg.mutate_params.sigma_frac}{COLORS['END']}
      Per-Gene Probability: {COLORS['RED']}{cfg.mutate_params.per_gene_prob}{COLORS['END']}
      Reset Probability: {COLORS['RED']}{cfg.mutate_params.reset_prob}{COLORS['END']}
    Crossover:
      Alpha Range: {COLORS['BLUE']}[{cfg.crossover_params.alpha_min}, {cfg.crossover_params.alpha_max}]{COLORS['END']}
      Per-Gene Probability: {COLORS['BLUE']}{cfg.crossover_params.per_gene_prob}{COLORS['END']}
    """
    print(config_str)

    # Run evolution
    runner = GARunner(cfg)
    color_print("=== Starting Evolution ===", "HEADER", bold=True)
    best_g, best_f = runner.evolve()

    # Results with colors
    color_print("\n=== Best Genome ===", "HEADER", bold=True)
    print(json.dumps(best_g.to_json(), indent=2))
    
    color_print("\n=== Best Fitness ===", "HEADER", bold=True)
    stats_str = (
        f"{COLORS['BOLD']}Median Score:{COLORS['END']} {COLORS['GREEN']}{best_f.median:.2f}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Win Rate:{COLORS['END']} {COLORS['CYAN']}{100*best_f.win_rate:.2f}%{COLORS['END']}\n"
        f"{COLORS['BOLD']}Max Score:{COLORS['END']} {COLORS['YELLOW']}{best_f.max_score}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Min Score:{COLORS['END']} {COLORS['RED']}{best_f.min_score}{COLORS['END']}\n"
        f"{COLORS['BOLD']}Bust Rate:{COLORS['END']} {COLORS['RED']}{100*best_f.diagnostics.get('bust_rate', 0):.2f}%{COLORS['END']}"
    )
    print(stats_str)