# ga/ga_runner.py
import json
import random
from typing import List, Tuple, Optional, Dict, Any

from .ga_types import GARunConfig, Fitness
from .ga_genome import Genome
from .ga_mutation import Mutator
from .ga_crossover import CrossoverOperator
from .ga_selection import FitnessComparator, Selector
from .ga_evaluator import PopulationEvaluator
from .ga_controls import ControlPool
from .ga_repository import BotRepository
from .ga_timing import StopWatch


class GARunner:
    """
    Orchestrates the evolutionary loop:
      - initialize population
      - evaluate (everyone plays together with control bots)
      - select elites + breed via tournament selection and crossover
      - mutate children
      - re-evaluate, track & save the best per generation
      - log timings + performance (with baseline win% multiplier)
    """

    def __init__(
        self,
        config: GARunConfig,
        evaluator: PopulationEvaluator | None = None,
        selector: Selector | None = None,
        mutator: Mutator | None = None,
        crossover: CrossoverOperator | None = None,
        repo: BotRepository | None = None,
    ):
        self.cfg = config
        self.evaluator = evaluator or PopulationEvaluator(ControlPool())
        self.selector = selector or Selector(FitnessComparator())
        self.mutator = mutator or Mutator(config.mutate_params)
        self.crossover = crossover or CrossoverOperator(config.crossover_params)
        self.repo = repo or BotRepository()

    # ---------- internal helpers ----------

    def _init_population(self, init_pop: Optional[List[Genome]]) -> List[Genome]:
        """Seed population; if no init_pop, mutate DEFAULT twice for diversity."""
        pop: List[Genome] = []
        if init_pop:
            pop = [g.clone() for g in init_pop]
        while len(pop) < self.cfg.pop_size:
            g = Genome.from_default()
            g = self.mutator.mutate(g)   # mutation pass 1
            g = self.mutator.mutate(g)   # mutation pass 2 (extra noise)
            pop.append(g)
        return pop

    def _current_baseline(self, pop_len: int) -> float:
        """Uniform-win baseline given table size (#pop + #controls)."""
        table_size = pop_len + len(self.evaluator.controls.make())
        return 1.0 / table_size if table_size > 0 else 0.0

    # ---------- main API ----------

    def evolve(self, init_pop: Optional[List[Genome]] = None) -> Tuple[Genome, Fitness]:
        # Initial setup + eval (timed)
        sw0 = StopWatch()
        pop = self._init_population(init_pop)
        sw0.tick("init_population")

        fits, ctrl = self.evaluator.evaluate_all_in_one(
            pop,
            games_per_eval=self.cfg.games_per_eval,
            base_seed=self.cfg.eval_seed,
            verbose_game=0,
        )
        sw0.tick("initial_evaluate")

        scored: List[Tuple[Genome, Fitness]] = list(zip(pop, fits))
        best_g, best_f = scored[0]
        for g, f in scored[1:]:
            if self.selector.comparator.better(f, best_f):
                best_g, best_f = g, f

        # One-time timing line
        print("[Timing] " + sw0.pretty_line("init"))

        # Pretty control print with baseline × multiplier
        def print_controls(ctrlfits: Dict[str, Fitness], baseline: float) -> None:
            parts = []
            for name in sorted(ctrlfits.keys()):
                f = ctrlfits[name]
                mult = (f.win_rate / baseline) if baseline > 0 else 0.0
                mean = f.diagnostics.get("mean", 0.0)
                sd = f.diagnostics.get("sd", 0.0)
                parts.append(
                    f"{name}: med={f.median:.2f}, "
                    f"avg={mean:.2f}, sd={sd:.2f}, "
                    f"win%={100*f.win_rate:.2f} ({mult:.2f}× base {100*baseline:.2f}%), "
                    f"max={f.max_score}, min={f.min_score}, "
                    f"bust%={100*f.diagnostics.get('bust_rate',0):.1f}"
                )
            print("Controls -> " + " | ".join(parts))

        # Generations
        for gen in range(self.cfg.generations):
            sw = StopWatch()

            # Elites
            elites = sorted(
                scored,
                key=lambda gf: (gf[1].median, gf[1].win_rate, gf[1].max_score, gf[1].min_score),
                reverse=True,
            )[: self.cfg.elitism]
            sw.tick("select_elites")

            # Breed (tournament selection → crossover → mutate)
            new_pop: List[Genome] = [e[0].clone() for e in elites]
            while len(new_pop) < self.cfg.pop_size:
                parents = self.selector.tournament(
                    [(g.to_json(), f) for g, f in scored],
                    tourney_size=self.cfg.tourney_size,
                    k=2,
                )
                pa = Genome(parents[0])
                pb = Genome(parents[1])
                child = self.crossover.crossover(pa, pb)
                if self.cfg.mutation_after_crossover:
                    child = self.mutator.mutate(child)
                new_pop.append(child)
            sw.tick("breed")

            # Evaluate new population (everyone + controls together)
            pop = new_pop
            fits, ctrl = self.evaluator.evaluate_all_in_one(
                pop,
                games_per_eval=self.cfg.games_per_eval,
                base_seed=self.cfg.eval_seed + gen + 1,
                verbose_game=0,
            )
            sw.tick("evaluate")

            scored = list(zip(pop, fits))

            # Best-of generation & global
            gen_best_g, gen_best_f = scored[0]
            for g, f in scored[1:]:
                if self.selector.comparator.better(f, gen_best_f):
                    gen_best_g, gen_best_f = g, f
            if self.selector.comparator.better(gen_best_f, best_f):
                best_g, best_f = gen_best_g, gen_best_f
            sw.tick("select_best")

            # Save best of this generation
            self.repo.save_best(
                gen_best_g.to_json(),
                gen_best_f,
                gen_idx=gen + 1,
                eval_seed=self.cfg.eval_seed + gen + 1,
            )
            sw.tick("save_best")

            # Logging with baseline + mean/sd
            baseline = self._current_baseline(len(pop))
            mult = (best_f.win_rate / baseline) if baseline > 0 else 0.0
            mean = best_f.diagnostics.get("mean", 0.0)
            sd = best_f.diagnostics.get("sd", 0.0)

            if (gen + 1) % self.cfg.log_every == 0:
                print(
                    f"[Gen {gen+1}/{self.cfg.generations}] "
                    f"best median={best_f.median:.2f} "
                    f"avg={mean:.2f}, sd={sd:.2f}, "
                    f"win%={100*best_f.win_rate:.2f} ({mult:.2f}× base {100*baseline:.2f}%) "
                    f"max={best_f.max_score} min={best_f.min_score} "
                    f"(bust%={100*best_f.diagnostics.get('bust_rate', 0):.2f}, "
                    f"games={self.cfg.games_per_eval}, pop={self.cfg.pop_size})"
                )
                print_controls(ctrl, baseline)

            # Per-gen timing line
            print("[Timing] " + sw.pretty_line(f"gen{gen+1}"))

        return best_g, best_f
