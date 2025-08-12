# evo_runner.py
import json
from ga.ga_types import GARunConfig, MutateParams, CrossoverParams
from ga.ga_runner import GARunner

if __name__ == "__main__":
    cfg = GARunConfig(
        pop_size=50,
        generations=10,
        games_per_eval=500,
        elitism=5,
        tourney_size=4,
        mutation_after_crossover=True,
        mutate_params=MutateParams(sigma_frac=0.12, per_gene_prob=0.35, reset_prob=0.06),
        crossover_params=CrossoverParams(alpha_min=0.3, alpha_max=0.7, per_gene_prob=0.9),
        eval_seed=42,
        log_every=1,
    )
    runner = GARunner(cfg)
    best_g, best_f = runner.evolve()

    print("\n=== Best genome ===")
    print(json.dumps(best_g.to_json(), indent=2))
    print("\n=== Best fitness ===")
    print(f"median={best_f.median:.2f}, win%={100*best_f.win_rate:.2f}, "
          f"max={best_f.max_score}, min={best_f.min_score}, "
          f"bust%={100*best_f.diagnostics.get('bust_rate', 0):.2f})")
