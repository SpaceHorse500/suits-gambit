# ga_runner.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from .ga_types import GARunConfig, Fitness
from .ga_genome import Genome
from .ga_mutation import Mutator
from .ga_crossover import Crosser
from .ga_evaluator import PopulationEvaluator
from .ga_controls import ControlPool


# -------------------------------
# Utilities
# -------------------------------

def short(uid: str, n: int = 8) -> str:
    return uid[:n]


def _extract_diag(f: Fitness) -> Tuple[float, float, float, float, float]:
    """
    Returns (median, q1, q3, sd, bust_rate) from Fitness.diagnostics,
    with robust fallbacks to older key names.
    """
    d = f.diagnostics or {}
    med = float(d.get("median", d.get("median_observed", 0.0)))
    q1 = float(d.get("q1", d.get("Q1", 0.0)))
    q3 = float(d.get("q3", d.get("Q3", 0.0)))
    sd = float(d.get("sd", d.get("stdev", 0.0)))
    bust = float(d.get("bust_rate", d.get("bust%", 0.0)))
    return med, q1, q3, sd, bust


def _score_key(f: Fitness) -> Tuple:
    """
    Sort: win_rate (desc), median (desc), Q1(desc), Q3(desc).
    """
    med, q1, q3, _, _ = _extract_diag(f)
    return (-f.win_rate, -med, -q1, -q3)


def _relative_factor(x: float, baseline: float) -> float:
    return (x / baseline) if baseline > 0 else 0.0


# -------------------------------
# GA Runner
# -------------------------------

@dataclass
class RunnerDeps:
    evaluator: PopulationEvaluator
    mutator: Mutator
    crosser: Crosser
    controls: ControlPool


class GARunner:
    """
    Orchestrates selection → reproduction → evaluation.
    Fix: provenance is now stored using the *final* child's uid (after mutation).
    Elite clones are logged as (self, self).
    """

    def __init__(self, cfg: GARunConfig, deps: RunnerDeps):
        self.cfg = cfg
        self.evaluator = deps.evaluator
        self.mutator = deps.mutator
        self.crosser = deps.crosser
        self.controls = deps.controls

        # Will be filled each generation for logging parentage:
        self._last_parent_map: Dict[str, Tuple[str, str]] = {}

        seed = cfg.seed if hasattr(cfg, "seed") else None
        self._rng = random.Random(seed) if seed is not None else random

    # ------------- Public -------------

    def evolve(self, init_pop: Optional[List[Genome]] = None) -> Tuple[Genome, Fitness]:
        t0 = time.time()

        pop = self._init_population(init_pop)
        print("\n=== Starting Evolution ===")
        # Initial eval (optional to show "INITIAL POPULATION" like your earlier logs)
        pop_f, ctrl_f, t_eval = self._evaluate(pop, gen_idx=0)
        self._print_population_stats(pop, pop_f, ctrl_f, header="INITIAL POPULATION")
        print(f"[Timing] init evaluation={t_eval:.2f}s | total={time.time()-t0:.2f}s\n")

        best_idx = self._argmin_by(pop_f, key=_score_key)  # actually argmin because key returns negatives
        best = pop[best_idx]
        best_fit = pop_f[best_idx]

        # Generations
        for gen in range(1, self.cfg.generations + 1):
            tg0 = time.time()

            # Selection + reproduction
            tsel0 = time.time()
            elites_idx = self._select_elites(pop_f, self.cfg.elitism)
            t_selection = time.time() - tsel0

            trep0 = time.time()
            pop = self._reproduce(pop, pop_f, elites_idx)  # updates self._last_parent_map
            t_repro = time.time() - trep0

            # Evaluate new pop
            pop_f, ctrl_f, t_eval = self._evaluate(pop, gen_idx=gen)

            # Track best
            gen_best_idx = self._argmin_by(pop_f, key=_score_key)
            if _score_key(pop_f[gen_best_idx]) < _score_key(best_fit):
                best = pop[gen_best_idx]
                best_fit = pop_f[gen_best_idx]

            # Logging
            print(f"=== Generation {gen}/{self.cfg.generations} ===\n")
            self._print_population_stats(pop, pop_f, ctrl_f)
            self._print_reproduction_summary(len(elites_idx), len(pop) - len(elites_idx))
            self._print_parentage(self._last_parent_map)
            self._print_controls(ctrl_f)
            self._print_top10_summary(pop, pop_f, ctrl_f)

            print(f"[Timing] gen{gen} selection={t_selection:.2f}s | reproduction={t_repro:.2f}s | "
                  f"evaluation={t_eval:.2f}s | total={time.time()-tg0:.2f}s\n")

        return best, best_fit

    # ------------- Core steps -------------

    def _init_population(self, init_pop: Optional[List[Genome]]) -> List[Genome]:
        if init_pop is not None and len(init_pop) > 0:
            return list(init_pop)

        P = self.cfg.population_size
        pop: List[Genome] = []
        # Create random genomes by mutating a base genome (from Genome.default() or similar)
        base = Genome.default() if hasattr(Genome, "default") else Genome()
        for _ in range(P):
            # Important: we only use the final uid (after mutation) everywhere
            g = self.mutator.mutate(base.clone())  # ok if mutate reassigns uid; we never stored it yet
            pop.append(g)
        return pop

    def _evaluate(self, pop: List[Genome], gen_idx: int) -> Tuple[List[Fitness], Dict[str, Fitness], float]:
        t0 = time.time()
        fits, ctrl = self.evaluator.evaluate_all_in_one(
            pop=pop,
            games_per_eval=self.cfg.games_per_eval,
            base_seed=(self.cfg.seed or 0) + gen_idx * 10_000,
            verbose_game=getattr(self.cfg, "verbose_game", 0),
        )
        return fits, ctrl, (time.time() - t0)

    def _select_elites(self, fits: List[Fitness], k: int) -> List[int]:
        idxs = list(range(len(fits)))
        idxs.sort(key=lambda i: _score_key(fits[i]))
        return idxs[:max(0, min(k, len(fits)))]

    def _tournament(self, pop: List[Genome], fits: List[Fitness], k: int) -> int:
        # Sample k indices, pick best by score key
        pool = self._rng.sample(range(len(pop)), k)
        pool.sort(key=lambda i: _score_key(fits[i]))
        return pool[0]

    def _reproduce(self, pop: List[Genome], fits: List[Fitness], elite_idx: List[int]) -> List[Genome]:
        P = len(pop)
        E = len(elite_idx)
        T = self.cfg.tournament_size

        # Keep elites (clone so they don't share same object; log them as (self,self))
        new_pop: List[Genome] = []
        parent_map: Dict[str, Tuple[str, str]] = {}

        for i in elite_idx:
            # If Genome.clone exists with keep_id flag, keep_id=False so clone gets a new uid.
            # We still tie provenance to the original elite (self, self).
            if hasattr(pop[i], "clone"):
                elite_clone = pop[i].clone()  # new uid
            else:
                # Fallback: shallow replacement
                elite_clone = Genome.from_json(pop[i].to_json()) if hasattr(Genome, "from_json") else pop[i]
            new_pop.append(elite_clone)
            parent_map[elite_clone.uid] = (pop[i].uid, pop[i].uid)

        # Create children to fill population
        needed = P - E
        for _ in range(needed):
            p1_idx = self._tournament(pop, fits, T)
            p2_idx = self._tournament(pop, fits, T)

            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            # crossover → mutate
            child = self.crosser.crossover(parent1, parent2, rng=self._rng) \
                if hasattr(self.crosser, "crossover") else parent1.clone()

            child = self.mutator.mutate(child, rng=self._rng)  # << child gets FINAL uid here

            # Store provenance **using the child's final uid**
            parent_map[child.uid] = (parent1.uid, parent2.uid)
            new_pop.append(child)

        # Save for logging
        self._last_parent_map = parent_map
        return new_pop

    # ------------- Logging -------------

    def _print_population_stats(self, pop: List[Genome], fits: List[Fitness],
                                ctrl_fits: Dict[str, Fitness],
                                header: Optional[str] = None) -> None:
        if header:
            print(f"=== {header} ===\n")
        print("Population Statistics:")
        # rank by score key
        order = list(range(len(pop)))
        order.sort(key=lambda i: _score_key(fits[i]))
        for rank, i in enumerate(order, 1):
            tag = "[ELITE]" if rank <= self.cfg.elitism else "[CHILD]"
            f = fits[i]
            med, q1, q3, sd, bust = _extract_diag(f)
            print(f" {rank:2d}. {tag} id={short(pop[i].uid)} "
                  f"win%={f.win_rate*100:.2f}% | "
                  f"med={med:.2f} [Q1={q1:.2f} | Q3={q3:.2f} | IQR={(q3-q1):.2f}] "
                  f"± {sd:.2f} | bust%={bust*100:.2f}%")
        print()

    def _print_reproduction_summary(self, elites: int, children: int) -> None:
        print("Reproduction Summary:")
        print(f"Elites carried over: {elites}")
        print(f"New children created: {children}")
        print(f"Genomes eliminated: {children - elites if children >= elites else 0}")
        print()

    def _print_parentage(self, parent_map: Dict[str, Tuple[str, str]]) -> None:
        print("Parent-Child Relationships:")
        # Only show children (i.e., entries where parents are not (self,self))
        lines = []
        for cid, (p1, p2) in parent_map.items():
            if p1 == p2 and short(cid)[:8] != short(p1)[:8]:
                # This is likely an elite clone; print anyway, but mark
                lines.append(f"{short(cid)} ← {short(p1)}, {short(p2)}")
            elif p1 == p2:
                # exact same id (extremely rare unless keep_id was used) – still print
                lines.append(f"{short(cid)} ← {short(p1)}, {short(p2)}")
            else:
                lines.append(f"{short(cid)} ← {short(p1)}, {short(p2)}")
        if not lines:
            print("(none)")
        else:
            # Show a stable subset for readability
            for ln in lines[: max(5, min(20, len(lines)))]:
                print(ln)
        print()

    def _print_controls(self, ctrl: Dict[str, Fitness]) -> None:
        # Baseline for “×” factor: average of all controls’ win%,
        # fall back to 1/num_players parity if empty
        if ctrl:
            base = sum(f.win_rate for f in ctrl.values()) / max(1, len(ctrl))
        else:
            base = 0.025  # safe tiny fallback

        print("Control Bots Performance:")
        for name, f in ctrl.items():
            med, q1, q3, sd, bust = _extract_diag(f)
            x = _relative_factor(f.win_rate, base)
            print(f"  {name}: win%={f.win_rate*100:.2f}% ({x:.2f}×) | "
                  f"med={med:.2f} [Q1={q1:.2f} | Q3={q3:.2f} | IQR={(q3-q1):.2f}] "
                  f"| bust%={bust*100:.2f}%")
        print()

    def _print_top10_summary(self, pop: List[Genome], fits: List[Fitness], ctrl: Dict[str, Fitness]) -> None:
        # Build combined list (genomes + controls), ranked by same key
        items: List[Tuple[str, str, str, Fitness]] = []  # (kind, label, uid, fit)

        # genomes
        for g, f in zip(pop, fits):
            items.append(("GEN", f"G:{short(g.uid)}", g.uid, f))

        # controls
        for name, f in ctrl.items():
            items.append(("CTRL", name, name, f))

        # rank
        items.sort(key=lambda x: _score_key(x[3]))

        # baseline for × factor: the average control win%
        if ctrl:
            base = sum(f.win_rate for f in ctrl.values()) / max(1, len(ctrl))
        else:
            base = 0.025

        print("=== SUMMARY: Top 10 (Genomes + Controls) ===")
        for rank, (kind, label, _, f) in enumerate(items[:10], 1):
            med, q1, q3, sd, bust = _extract_diag(f)
            x = _relative_factor(f.win_rate, base)
            print(f" {rank}. [{kind}] {label}: "
                  f"win%={f.win_rate*100:.2f}% ({x:.2f}×) | "
                  f"med={med:.2f} [Q1={q1:.2f} | Q3={q3:.2f} | IQR={(q3-q1):.2f}] | "
                  f"± {sd:.2f} | bust%={bust*100:.2f}%")
        print()

    # ------------- helpers -------------

    @staticmethod
    def _argmin_by(xs: Iterable, key):
        best_i = None
        best_k = None
        for i, x in enumerate(xs):
            k = key(x)
            if best_i is None or k < best_k:
                best_i, best_k = i, k
        return best_i
