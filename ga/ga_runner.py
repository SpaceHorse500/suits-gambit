# ga_runner.py
import json, random
from typing import List, Tuple, Optional, Dict, Any
from .ga_types import GARunConfig, Fitness
from .ga_genome import Genome
from .ga_mutation import Mutator
from .ga_crossover import CrossoverOperator
from .ga_selection import FitnessComparator, Selector
from .ga_evaluator import PopulationEvaluator
from .ga_controls import ControlPool
from .ga_repository import BotRepository
from evo_io import DEFAULT_GENOME

class GARunner:
    def __init__(self,
                 config: GARunConfig,
                 evaluator: PopulationEvaluator | None = None,
                 selector: Selector | None = None,
                 mutator: Mutator | None = None,
                 crossover: CrossoverOperator | None = None,
                 repo: BotRepository | None = None):
        self.cfg = config
        self.evaluator = evaluator or PopulationEvaluator(ControlPool())
        self.selector = selector or Selector(FitnessComparator())
        self.mutator = mutator or Mutator(config.mutate_params)
        self.crossover = crossover or CrossoverOperator(config.crossover_params)
        self.repo = repo or BotRepository()

    def _init_population(self, init_pop: Optional[List[Genome]]) -> List[Genome]:
        rng = random.Random(self.cfg.eval_seed)
        pop: List[Genome] = []
        if init_pop:
            pop = [g.clone() for g in init_pop]
        while len(pop) < self.cfg.pop_size:
            # seed diversity: mutate default with strong params
            g = Genome.from_default()
            g = self.mutator.mutate(g)
            # add extra noise
            m2 = Mutator(self.cfg.mutate_params)
            g = m2.mutate(g)
            pop.append(g)
        return pop

    def evolve(self, init_pop: Optional[List[Genome]] = None) -> Tuple[Genome, Fitness]:
        pop = self._init_population(init_pop)

        fits, ctrl = self.evaluator.evaluate_all_in_one(
            pop, games_per_eval=self.cfg.games_per_eval, base_seed=self.cfg.eval_seed, verbose_game=0
        )
        scored: List[Tuple[Genome, Fitness]] = list(zip(pop, fits))

        best_g, best_f = scored[0]
        for g, f in scored[1:]:
            if self.selector.comparator.better(f, best_f):
                best_g, best_f = g, f

        # logging helper
        def print_controls(ctrlfits: Dict[str, Fitness]):
            parts = []
            for name in sorted(ctrlfits.keys()):
                f = ctrlfits[name]
                parts.append(f"{name}: med={f.median:.2f}, win%={100*f.win_rate:.2f}, max={f.max_score}, min={f.min_score}, bust%={100*f.diagnostics.get('bust_rate',0):.1f}")
            print("Controls -> " + " | ".join(parts))

        for gen in range(self.cfg.generations):
            # elites
            elites = sorted(scored, key=lambda gf: (
                gf[1].median, gf[1].win_rate, gf[1].max_score, gf[1].min_score
            ), reverse=True)[:self.cfg.elitism]

            # breed
            new_pop: List[Genome] = [e[0].clone() for e in elites]
            while len(new_pop) < self.cfg.pop_size:
                parents = self.selector.tournament([(g.to_json(), f) for g, f in scored],
                                                   tourney_size=self.cfg.tourney_size, k=2)
                pa = Genome(parents[0]); pb = Genome(parents[1])
                child = self.crossover.crossover(pa, pb)
                if self.cfg.mutation_after_crossover:
                    child = self.mutator.mutate(child)
                new_pop.append(child)

            pop = new_pop
            fits, ctrl = self.evaluator.evaluate_all_in_one(
                pop, games_per_eval=self.cfg.games_per_eval, base_seed=self.cfg.eval_seed + gen + 1, verbose_game=0
            )
            scored = list(zip(pop, fits))

            # update best
            gen_best_g, gen_best_f = scored[0]
            for g, f in scored[1:]:
                if self.selector.comparator.better(f, gen_best_f):
                    gen_best_g, gen_best_f = g, f
            if self.selector.comparator.better(gen_best_f, best_f):
                best_g, best_f = gen_best_g, gen_best_f

            # save + log
            if (gen + 1) % self.cfg.log_every == 0:
                print(f"[Gen {gen+1}/{self.cfg.generations}] "
                      f"best median={best_f.median:.2f} win%={100*best_f.win_rate:.2f} "
                      f"max={best_f.max_score} min={best_f.min_score} "
                      f"(bust%={100*best_f.diagnostics.get('bust_rate', 0):.2f}, "
                      f"games={self.cfg.games_per_eval}, pop={self.cfg.pop_size})")
                print_controls(ctrl)

            # save best of THIS generation
            self.repo.save_best(gen_best_g.to_json(), gen_best_f, gen_idx=gen+1, eval_seed=self.cfg.eval_seed + gen + 1)

        return best_g, best_f
