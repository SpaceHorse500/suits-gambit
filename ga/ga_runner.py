# ga/ga_runner.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from .ga_types import GARunConfig, Fitness
from .ga_genome import Genome
from .ga_mutation import Mutator
from .ga_crossover import CrossoverOperator
from .ga_evaluator import PopulationEvaluator
from .ga_controls import ControlPool
from .ga_repository import BotRepository  # <-- NEW

# -------------------------------
# ANSI colors
# -------------------------------

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

def color_text(text: str, color: str = "GREEN", bold: bool = False) -> str:
    style = COLORS["BOLD"] if bold else ""
    return f"{style}{COLORS.get(color,'')}{text}{COLORS['END']}"

# -------------------------------
# Utilities
# -------------------------------

def short(uid: str, n: int = 8) -> str:
    return uid[:n]

def _extract_diag(f: Fitness) -> Tuple[float, float, float, float, float]:
    """Returns (median, q1, q3, stdev, bust_rate) from either top-level Fitness fields
    or f.diagnostics (with several alias names)."""
    d = getattr(f, "diagnostics", None) or {}
    med  = _pick(f, d, 0.0, "median", "median_observed", "med")
    q1   = _pick(f, d, 0.0, "q1", "Q1", "p25")
    q3   = _pick(f, d, 0.0, "q3", "Q3", "p75")
    sd   = _pick(f, d, 0.0, "sd", "stdev", "std")
    bust = _pick(f, d, 0.0, "bust_rate", "bust%", "bust")
    return med, q1, q3, sd, bust

def _pick(obj, diag: dict, default: float, *names: str) -> float:
    # 1) object attribute
    for nm in names:
        if hasattr(obj, nm):
            v = getattr(obj, nm)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
    # 2) diagnostics dict
    for nm in names:
        if nm in diag:
            try:
                return float(diag[nm])
            except Exception:
                pass
    return float(default)

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
    crosser: CrossoverOperator
    controls: ControlPool
    repo: Optional[BotRepository] = None  # <-- NEW

class GARunner:
    """
    Selection → reproduction → evaluation with colorful, detailed reporting.
    Also saves the Top-1 (non-control) genome each generation via BotRepository.
    """

    def __init__(self, cfg: GARunConfig, deps: RunnerDeps):
        self.cfg = cfg
        self.evaluator = deps.evaluator
        self.mutator = deps.mutator
        self.crosser = deps.crosser
        self.controls = deps.controls
        self.repo: Optional[BotRepository] = deps.repo or BotRepository()  # <-- NEW

        # Filled each generation for logging parentage:
        self._last_parent_map: Dict[str, Tuple[str, str]] = {}

        seed = cfg.seed if hasattr(cfg, "seed") else getattr(cfg, "eval_seed", None)
        self._rng = random.Random(seed) if seed is not None else random

    # ------------- Public -------------

    def evolve(self, init_pop: Optional[List[Genome]] = None) -> Tuple[Genome, Fitness]:
        t0 = time.time()

        pop = self._init_population(init_pop)
        print(color_text("\n=== INITIAL POPULATION ===", "HEADER", bold=True))
        pop_f, ctrl_f, t_eval = self._evaluate(pop, gen_idx=0)
        self._print_population_stats(pop, pop_f, ctrl_f, header=None)
        self._print_controls(ctrl_f)
        self._print_top10_summary(pop, pop_f, ctrl_f)
        print(color_text(f"[Timing] init evaluation={t_eval:.2f}s | total={time.time()-t0:.2f}s\n", "BLUE"))

        # Best of generation 0 (non-controls)
        best_idx = self._argmin_by(pop_f, key=_score_key)  # argmin because key returns negatives
        best = pop[best_idx]
        best_fit = pop_f[best_idx]

        # SAVE Top-1 of gen 0
        self._save_top1(gen_idx=0, genome=best, fitness=best_fit)

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

            # Top-of-generation (non-controls)
            gen_best_idx = self._argmin_by(pop_f, key=_score_key)

            # Update global best
            if _score_key(pop_f[gen_best_idx]) < _score_key(best_fit):
                best = pop[gen_best_idx]
                best_fit = pop_f[gen_best_idx]

            # SAVE Top-1 of this generation
            self._save_top1(gen_idx=gen, genome=pop[gen_best_idx], fitness=pop_f[gen_best_idx])

            # Logging
            print(color_text(f"=== Generation {gen}/{self.cfg.generations} ===\n", "HEADER", bold=True))
            self._print_population_stats(pop, pop_f, ctrl_f)
            self._print_reproduction_summary(len(elites_idx), len(pop) - len(elites_idx))
            self._print_parentage(self._last_parent_map)
            self._print_controls(ctrl_f)
            self._print_top10_summary(pop, pop_f, ctrl_f)

            print(color_text(
                f"[Timing] gen{gen} selection={t_selection:.2f}s | reproduction={t_repro:.2f}s | "
                f"evaluation={t_eval:.2f}s | total={time.time()-tg0:.2f}s\n",
                "BLUE",
            ))

        return best, best_fit

    # ------------- Core steps -------------

    def _init_population(self, init_pop: Optional[List[Genome]]) -> List[Genome]:
        if init_pop is not None and len(init_pop) > 0:
            return list(init_pop)

        P = getattr(self.cfg, "population_size", None) or getattr(self.cfg, "pop_size")
        pop: List[Genome] = []
        base = Genome.default() if hasattr(Genome, "default") else Genome()
        for _ in range(P):
            g = self.mutator.mutate(base.clone(), rng=self._rng)  # use runner RNG to diversify
            pop.append(g)
        return pop

    def _evaluate(self, pop: List[Genome], gen_idx: int) -> Tuple[List[Fitness], Dict[str, Fitness], float]:
        t0 = time.time()
        games_per = getattr(self.cfg, "games_per_eval", 1000)
        seed_base = (getattr(self.cfg, "seed", None) or getattr(self.cfg, "eval_seed", 0)) + gen_idx * 10_000
        verbose = getattr(self.cfg, "verbose_game", 0)
        fits, ctrl = self.evaluator.evaluate_all_in_one(
            pop=pop,
            games_per_eval=games_per,
            base_seed=seed_base,
            verbose_game=verbose,
        )
        return fits, ctrl, (time.time() - t0)

    def _select_elites(self, fits: List[Fitness], k: int) -> List[int]:
        idxs = list(range(len(fits)))
        idxs.sort(key=lambda i: _score_key(fits[i]))
        return idxs[:max(0, min(k, len(fits)))]

    def _tournament(self, pop: List[Genome], fits: List[Fitness], k: int) -> int:
        pool = self._rng.sample(range(len(pop)), k)
        pool.sort(key=lambda i: _score_key(fits[i]))
        return pool[0]

    def _reproduce(self, pop: List[Genome], fits: List[Fitness], elite_idx: List[int]) -> List[Genome]:
        P = len(pop)
        E = len(elite_idx)
        T = getattr(self.cfg, "tournament_size", getattr(self.cfg, "tourney_size", 8))

        new_pop: List[Genome] = []
        parent_map: Dict[str, Tuple[str, str]] = {}

        # Elites
        for i in elite_idx:
            elite_clone = pop[i].clone() if hasattr(pop[i], "clone") else Genome.from_json(pop[i].to_json())
            new_pop.append(elite_clone)
            parent_map[getattr(elite_clone, "uid", elite_clone.id)] = (
                getattr(pop[i], "uid", pop[i].id),
                getattr(pop[i], "uid", pop[i].id),
            )

        # Children
        needed = P - E
        for _ in range(needed):
            p1_idx = self._tournament(pop, fits, T)
            p2_idx = self._tournament(pop, fits, T)
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            child = self.crosser.crossover(parent1, parent2, rng=self._rng) \
                if hasattr(self.crosser, "crossover") else parent1.clone()
            child = self.mutator.mutate(child, rng=self._rng)

            parent_map[getattr(child, "uid", child.id)] = (
                getattr(parent1, "uid", parent1.id),
                getattr(parent2, "uid", parent2.id),
            )
            new_pop.append(child)

        self._last_parent_map = parent_map
        return new_pop

    # ------------- Saving -------------

    def _save_top1(self, gen_idx: int, genome: Genome, fitness: Fitness) -> None:
        if not self.repo:
            return
        try:
            bot_id = getattr(genome, "uid", getattr(genome, "id", None))
            eval_seed = getattr(self.cfg, "seed", None) or getattr(self.cfg, "eval_seed", 0)
            payload = genome.to_json() if hasattr(genome, "to_json") else getattr(genome, "data", {})
            self.repo.save_best(
                genome_dict=payload,
                fitness=fitness,
                gen_idx=gen_idx,
                eval_seed=eval_seed,
                bot_id=bot_id,
            )
        except Exception as e:
            print(color_text(f"[WARN] Failed to save Top-1 for gen {gen_idx}: {e}", "YELLOW"))

    # ------------- Logging -------------

    def _print_population_stats(
        self,
        pop: List[Genome],
        fits: List[Fitness],
        ctrl_fits: Dict[str, Fitness],
        header: Optional[str] = None
    ) -> None:
        if header:
            print(color_text(f"=== {header} ===\n", "HEADER", bold=True))

        print(color_text("Population Statistics:", "BLUE", bold=True))
        order = list(range(len(pop)))
        order.sort(key=lambda i: _score_key(fits[i]))
        for rank, i in enumerate(order, 1):
            tag = color_text("[ELITE]", "GREEN") if rank <= getattr(self.cfg, "elitism", 0) else color_text("[CHILD]", "CYAN")
            f = fits[i]
            med, q1, q3, sd, bust = _extract_diag(f)

            win_color  = "GREEN" if f.win_rate >= 0.05 else ("YELLOW" if f.win_rate >= 0.03 else "RED")
            bust_color = "RED" if bust > 0.55 else ("YELLOW" if bust > 0.45 else "GREEN")

            print(
                f" {rank:2d}. {tag} id={color_text(short(getattr(pop[i],'uid', pop[i].id)), 'CYAN')} "
                f"win%={color_text(f'{f.win_rate*100:.2f}%', win_color)} | "
                f"med={color_text(f'{med:.2f}', 'CYAN')} [Q1={color_text(f'{q1:.2f}','CYAN')} | "
                f"Q3={color_text(f'{q3:.2f}','CYAN')} | IQR={color_text(f'{(q3-q1):.2f}','CYAN')}] "
                f"± {sd:.2f} | bust%={color_text(f'{bust*100:.2f}%', bust_color)}"
            )
        print()

    def _print_reproduction_summary(self, elites: int, children: int) -> None:
        print(color_text("Reproduction Summary:", "BLUE", bold=True))
        print(f"Elites carried over: {color_text(str(elites), 'GREEN')}")
        print(f"New children created: {color_text(str(children), 'CYAN')}")
        print(f"Genomes eliminated: {color_text(str(children - elites if children >= elites else 0), 'RED')}")
        print()

    def _print_parentage(self, parent_map: Dict[str, Tuple[str, str]]) -> None:
        print(color_text("Parent-Child Relationships:", "BLUE", bold=True))
        lines = []
        for cid, (p1, p2) in parent_map.items():
            lines.append(f"{color_text(short(cid),'CYAN')} ← {color_text(short(p1),'YELLOW')}, {color_text(short(p2),'YELLOW')}")
        if not lines:
            print("(none)")
        else:
            for ln in lines[: max(5, min(20, len(lines)))]:
                print(ln)
        print()

    def _print_controls(self, ctrl: Dict[str, Fitness]) -> None:
        # Baseline for “×” factor: average of all controls’ win%
        base = (sum(f.win_rate for f in ctrl.values()) / max(1, len(ctrl))) if ctrl else 0.025
        print(color_text("Control Bots Performance:", "BLUE", bold=True))
        for name, f in ctrl.items():
            med, q1, q3, sd, bust = _extract_diag(f)
            x = _relative_factor(f.win_rate, base)
            bust_color = "RED" if bust > 0.55 else ("YELLOW" if bust > 0.45 else "GREEN")
            print(
                f"  {color_text(name, 'BLUE')}: "
                f"win%={color_text(f'{f.win_rate*100:.2f}%', 'GREEN')} ({color_text(f'{x:.2f}×', 'GREEN' if x>=1.0 else 'RED')}) | "
                f"med={color_text(f'{med:.2f}','CYAN')} [Q1={color_text(f'{q1:.2f}','CYAN')} | Q3={color_text(f'{q3:.2f}','CYAN')} | "
                f"IQR={color_text(f'{(q3-q1):.2f}','CYAN')}] | bust%={color_text(f'{bust*100:.2f}%', bust_color)}"
            )
        print()

    def _print_top10_summary(self, pop: List[Genome], fits: List[Fitness], ctrl: Dict[str, Fitness]) -> None:
        items: List[Tuple[str, str, str, Fitness]] = []  # (kind, label, uid, fit)
        for g, f in zip(pop, fits):
            items.append(("GEN", f"G:{short(getattr(g,'uid', g.id))}", getattr(g,'uid', g.id), f))
        for name, f in ctrl.items():
            items.append(("CTRL", name, name, f))
        items.sort(key=lambda x: _score_key(x[3]))

        base = (sum(f.win_rate for f in ctrl.values()) / max(1, len(ctrl))) if ctrl else 0.025

        print(color_text("=== SUMMARY: Top 10 (Genomes + Controls) ===", "HEADER", bold=True))
        for rank, (kind, label, _, f) in enumerate(items[:10], 1):
            med, q1, q3, sd, bust = _extract_diag(f)
            x = _relative_factor(f.win_rate, base)
            win_color  = "GREEN" if f.win_rate >= 0.05 else ("YELLOW" if f.win_rate >= 0.03 else "RED")
            bust_color = "RED" if bust > 0.55 else ("YELLOW" if bust > 0.45 else "GREEN")
            print(
                f" {rank:2d}. [{kind}] {color_text(label,'CYAN')}: "
                f"win%={color_text(f'{f.win_rate*100:.2f}%', win_color)} ({color_text(f'{x:.2f}×','GREEN' if x>=1 else 'RED')}) | "
                f"med={color_text(f'{med:.2f}','CYAN')} [Q1={color_text(f'{q1:.2f}','CYAN')} | Q3={color_text(f'{q3:.2f}','CYAN')} | "
                f"IQR={color_text(f'{(q3-q1):.2f}','CYAN')}] | ± {sd:.2f} | bust%={color_text(f'{bust*100:.2f}%', bust_color)}"
            )
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
