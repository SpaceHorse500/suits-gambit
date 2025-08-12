# eva_ga.py
from __future__ import annotations
import copy
import json
import random
import statistics as stats
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

from evo_io import DEFAULT_GENOME
from evo_player import EvoPlayer

# Controls / engine
from random_player import RandomPlayer
from smart_player import SmartPlayer
from hand_player import HandPlayer
from game import SuitsGambitGame


# =========================
# Bounds & typing hints
# =========================
FLOAT_BOUNDS: Dict[str, Tuple[float, float]] = {
    # forbidden logits
    "forbidden.w_stats": (-3.0, 3.0),
    "forbidden.w_match": (-3.0, 3.0),
    "forbidden.w_random": (-3.0, 3.0),
    "forbidden.w_anti": (-3.0, 3.0),
    "forbidden.prefer_info_if_tied": (0.0, 1.0),
    "forbidden.gate_last_seat_boost_stats": (-1.0, 1.5),
    "forbidden.gate_trailing_boost_stats": (-1.0, 1.5),
    "forbidden.trail_threshold": (0.0, 12.0),

    # draw policy
    "draw_policy.plus_scale_base": (0.5, 1.4),
    "draw_policy.times_scale_base": (0.4, 1.4),
    "draw_policy.w_scale_lead": (-0.2, 0.2),
    "draw_policy.w_scale_trail": (-0.2, 0.2),
    "draw_policy.w_scale_is_last": (-0.2, 0.2),
    "draw_policy.w_scale_round_weight": (-0.1, 0.1),
    "draw_policy.w_scale_suit_skew": (-0.2, 0.2),
    "draw_policy.w_scale_overtake": (-0.2, 0.2),
    "draw_policy.jitter": (0.0, 0.06),
    "draw_policy.push_times_to3_if_p_bust_below": (0.05, 0.50),
    "draw_policy.push_times_to4_if_p_bust_below": (0.05, 0.40),
    "draw_policy.push_plus_to4_if_p_bust_below": (0.05, 0.40),
    "draw_policy.brake_if_large_lead_r5": (0.0, 20.0),
    "draw_policy.brake_scale_r5": (0.0, 0.5),
    "draw_policy.bank5_bias_plus": (0.0, 0.9),
    "draw_policy.bank5_bias_times": (0.0, 0.9),

    # ops policy
    "ops_policy.x_prob_s_le2": (0.0, 0.9),
    "ops_policy.x_prob_s_3_4": (0.0, 0.9),
    "ops_policy.x_prob_s_ge5": (0.0, 0.95),
    "ops_policy.w_xprob_trail": (-0.3, 0.3),
    "ops_policy.w_xprob_lead": (-0.3, 0.3),
    "ops_policy.w_xprob_is_last": (-0.3, 0.3),
    "ops_policy.w_xprob_round": (-0.3, 0.3),
    "ops_policy.x_repeat_s_le3": (0.0, 0.9),
    "ops_policy.x_repeat_s_4_5": (0.0, 0.9),
    "ops_policy.x_repeat_s_ge6": (0.0, 0.95),
    "ops_policy.w_xrep_trail": (-0.3, 0.3),
    "ops_policy.w_xrep_lead": (-0.3, 0.3),
    "ops_policy.w_xrep_round": (-0.3, 0.3),
    "ops_policy.w_xprob_overtake": (-0.3, 0.3),
    "ops_policy.w_xrep_overtake": (-0.3, 0.3),

    # overtake proxy
    "overtake_proxy.w_trail": (0.0, 1.5),
    "overtake_proxy.w_rounds_left": (0.0, 1.5),
    "overtake_proxy.w_is_last": (0.0, 1.5),

    # fitness penalty (kept for future use)
    "fitness_penalty.seat_variance_weight": (0.0, 0.5),
}

INT_KEYS = {
    "draw_policy.min_target_plus": (2, 6),
    "draw_policy.min_target_times": (2, 6),
    "draw_policy.min_target_last_seat": (-1, 2),
    "draw_policy.min_target_r5_bump": (-1, 2),
}

def _iter_paths(d: Dict[str, Any], prefix: str = "") -> List[str]:
    out = []
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out += _iter_paths(v, p)
        else:
            out.append(p)
    return out

def _get(d: Dict[str, Any], path: str):
    cur = d
    ks = path.split(".")
    for k in ks[:-1]:
        cur = cur[k]
    return cur, ks[-1]

def _clamp_to_bounds(path: str, val):
    if path in INT_KEYS:
        lo, hi = INT_KEYS[path]
        return int(max(lo, min(hi, round(val))))
    if path in FLOAT_BOUNDS:
        lo, hi = FLOAT_BOUNDS[path]
        return max(lo, min(hi, float(val)))
    return val


# =========================
# Mutation & crossover
# =========================
@dataclass
class MutateParams:
    sigma_frac: float = 0.10     # Gaussian nudge ~10% of range
    per_gene_prob: float = 0.30  # mutate chance per gene
    reset_prob: float = 0.05     # random reset instead of Gaussian
    seed: Optional[int] = None

def _range_for(path: str) -> Tuple[float, float]:
    if path in INT_KEYS:
        lo, hi = INT_KEYS[path]
        return float(lo), float(hi)
    if path in FLOAT_BOUNDS:
        return FLOAT_BOUNDS[path]
    return -1.0, 1.0

def mutate_genome(g: Dict[str, Any], mp: MutateParams = MutateParams()) -> Dict[str, Any]:
    rnd = random.Random(mp.seed) if mp.seed is not None else random
    child = copy.deepcopy(g)
    for path in _iter_paths(child):
        holder, key = _get(child, path)
        val = holder[key]
        if not isinstance(val, (int, float)):
            continue
        if rnd.random() > mp.per_gene_prob:
            continue

        lo, hi = _range_for(path)
        span = hi - lo if hi > lo else 1.0

        if rnd.random() < mp.reset_prob:
            if path in INT_KEYS:
                holder[key] = int(rnd.randint(int(lo), int(hi)))
            else:
                holder[key] = _clamp_to_bounds(path, lo + rnd.random() * span)
            continue

        sigma = mp.sigma_frac * span
        nudged = val + rnd.gauss(0.0, sigma)
        holder[key] = _clamp_to_bounds(path, nudged)

    return child

@dataclass
class CrossoverParams:
    alpha_min: float = 0.25
    alpha_max: float = 0.75
    per_gene_prob: float = 0.90
    seed: Optional[int] = None

def crossover(a: Dict[str, Any], b: Dict[str, Any], cp: CrossoverParams = CrossoverParams()) -> Dict[str, Any]:
    rnd = random.Random(cp.seed) if cp.seed is not None else random
    child = copy.deepcopy(a)
    for path in _iter_paths(child):
        if rnd.random() > cp.per_gene_prob:
            continue
        hold_c, key_c = _get(child, path)
        hold_a, key_a = _get(a, path)
        hold_b, key_b = _get(b, path)
        va = hold_a[key_a]
        vb = hold_b[key_b]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            alpha = rnd.uniform(cp.alpha_min, cp.alpha_max)
            val = alpha * va + (1 - alpha) * vb
            if path in INT_KEYS:
                val = round(val)
            hold_c[key_c] = _clamp_to_bounds(path, val)
        else:
            hold_c[key_c] = copy.deepcopy(va)
    return child


# =========================
# Fitness & evaluation
# =========================
@dataclass
class Fitness:
    median: float
    win_rate: float
    max_score: int
    min_score: int
    diagnostics: Dict[str, Any]

def lexicographic_better(a: Fitness, b: Fitness,
                         median_tol: float = 0.2,
                         win_tol: float = 0.005) -> bool:
    if abs(a.median - b.median) > median_tol:
        return a.median > b.median
    if abs(a.win_rate - b.win_rate) > win_tol:
        return a.win_rate > b.win_rate
    if a.max_score != b.max_score:
        return a.max_score > b.max_score
    if a.min_score != b.min_score:
        return a.min_score > b.min_score
    # soft tie-break: lower bust rate
    ba = a.diagnostics.get("bust_rate", 1.0)
    bb = b.diagnostics.get("bust_rate", 1.0)
    return ba < bb

def _control_bots() -> List:
    # Exactly the controls you asked for
    return [
        RandomPlayer("Random2"),
        SmartPlayer("Smart1"),
        HandPlayer("Hand1"),
    ]

def evaluate_population_all_in_one(pop_genomes: List[Dict[str, Any]],
                                   games_per_eval: int = 1000,
                                   base_seed: int = 123,
                                   verbose_game: int = 0
                                   ) -> Tuple[List[Fitness], Dict[str, Fitness]]:
    """
    Everyone plays together in the same table each game:
    [Evo0..EvoP-1] + controls(Random2, Smart1, Hand1).
    Returns:
      - Fitness list for each genome (aligned with pop_genomes order)
      - Dict[str, Fitness] for control bots
    """
    P = len(pop_genomes)
    control_names = [b.name for b in _control_bots()]
    table_size = P + len(control_names)
    assert table_size >= 2, "Need at least two players"

    # per-genome accumulators
    totals = [[] for _ in range(P)]
    wins = [0 for _ in range(P)]
    bust_rounds = [0 for _ in range(P)]
    rounds_played = [0 for _ in range(P)]
    max_score = [0 for _ in range(P)]
    min_score = [None for _ in range(P)]

    # per-control accumulators
    ctrl_totals: Dict[str, List[int]] = {n: [] for n in control_names}
    ctrl_wins: Dict[str, int] = {n: 0 for n in control_names}
    ctrl_bust_rounds: Dict[str, int] = {n: 0 for n in control_names}
    ctrl_rounds_played: Dict[str, int] = {n: 0 for n in control_names}
    ctrl_max: Dict[str, int] = {n: 0 for n in control_names}
    ctrl_min: Dict[str, Optional[int]] = {n: None for n in control_names}

    for gidx in range(games_per_eval):
        evos = [EvoPlayer(f"Evo{i}", genome=pop_genomes[i]) for i in range(P)]
        players = evos + _control_bots()
        random.Random(base_seed + gidx).shuffle(players)

        game = SuitsGambitGame(players, verbose=verbose_game, seed=(base_seed + gidx))
        winner, results = game.play()

        # map names back
        name_to_idx = {f"Evo{i}": i for i in range(P)}

        for p in players:
            score = results[p.name]
            if p.name in name_to_idx:
                i = name_to_idx[p.name]
                totals[i].append(score)
                max_score[i] = max(max_score[i], score)
                min_score[i] = score if min_score[i] is None else min(min_score[i], score)
                rounds_played[i] += len(p.round_scores)
                bust_rounds[i] += sum(1 for s in p.round_scores if s == 0)
            elif p.name in ctrl_totals:
                ctrl_totals[p.name].append(score)
                ctrl_max[p.name] = max(ctrl_max[p.name], score)
                ctrl_min[p.name] = score if ctrl_min[p.name] is None else min(ctrl_min[p.name], score)
                ctrl_rounds_played[p.name] += len(p.round_scores)
                ctrl_bust_rounds[p.name] += sum(1 for s in p.round_scores if s == 0)

        if winner in name_to_idx:
            wins[name_to_idx[winner]] += 1
        elif winner in ctrl_wins:
            ctrl_wins[winner] += 1

    # finalize genomes
    fits: List[Fitness] = []
    for i in range(P):
        ts = totals[i]
        if not ts:
            fits.append(Fitness(0.0, 0.0, 0, 0, {"bust_rate": 1.0, "mean": 0.0, "sd": 0.0, "n_games": 0}))
            continue
        med = float(stats.median(ts))
        mx = int(max_score[i])
        mn = int(min_score[i]) if min_score[i] is not None else 0
        wr = wins[i] / max(1, games_per_eval)
        br = (bust_rounds[i] / rounds_played[i]) if rounds_played[i] else 1.0
        fits.append(Fitness(
            median=med,
            win_rate=wr,
            max_score=mx,
            min_score=mn,
            diagnostics={
                "bust_rate": br,
                "mean": (sum(ts) / len(ts)),
                "sd": (stats.pstdev(ts) if len(ts) > 1 else 0.0),
                "n_games": len(ts),
            }
        ))

    # finalize controls
    control_fits: Dict[str, Fitness] = {}
    for n in control_names:
        ts = ctrl_totals[n]
        if not ts:
            control_fits[n] = Fitness(0.0, 0.0, 0, 0, {"bust_rate": 1.0, "mean": 0.0, "sd": 0.0, "n_games": 0})
            continue
        med = float(stats.median(ts))
        mx = int(ctrl_max[n])
        mn = int(ctrl_min[n]) if ctrl_min[n] is not None else 0
        wr = ctrl_wins[n] / max(1, games_per_eval)
        rounds = ctrl_rounds_played[n]
        br = (ctrl_bust_rounds[n] / rounds) if rounds else 1.0
        control_fits[n] = Fitness(
            median=med,
            win_rate=wr,
            max_score=mx,
            min_score=mn,
            diagnostics={
                "bust_rate": br,
                "mean": (sum(ts) / len(ts)),
                "sd": (stats.pstdev(ts) if len(ts) > 1 else 0.0),
                "n_games": len(ts),
            }
        )

    return fits, control_fits


# =========================
# Selection helpers
# =========================
def tournament_select(pop: List[Tuple[Dict[str, Any], Fitness]],
                      tourney_size: int = 4,
                      k: int = 1,
                      rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    rng = rng or random
    selected: List[Dict[str, Any]] = []
    for _ in range(k):
        group = rng.sample(pop, tourney_size)
        best_g, best_f = group[0]
        for g, f in group[1:]:
            if lexicographic_better(f, best_f):
                best_g, best_f = g, f
        selected.append(copy.deepcopy(best_g))
    return selected


# =========================
# GA runner
# =========================
@dataclass
class GARunConfig:
    pop_size: int = 50                       # everyone plays together
    generations: int = 20
    games_per_eval: int = 500               # total shared games per generation
    elitism: int = 5
    tourney_size: int = 4
    mutation_after_crossover: bool = True
    mutate_params: MutateParams = field(default_factory=MutateParams)
    crossover_params: CrossoverParams = field(default_factory=CrossoverParams)
    eval_seed: int = 123
    log_every: int = 1

def _print_controls_summary(ctrl: Dict[str, Fitness]) -> None:
    names = sorted(ctrl.keys())
    line = " | ".join(
        f"{n}: med={ctrl[n].median:.2f}, win%={100*ctrl[n].win_rate:.2f}, "
        f"max={ctrl[n].max_score}, min={ctrl[n].min_score}, "
        f"bust%={100*ctrl[n].diagnostics.get('bust_rate',0):.1f}"
        for n in names
    )
    print("Controls -> " + line)

def evolve(config: GARunConfig,
           init_pop: Optional[List[Dict[str, Any]]] = None,
           rng: Optional[random.Random] = None) -> Tuple[Dict[str, Any], Fitness]:
    rng = rng or random

    # init population
    pop: List[Dict[str, Any]] = []
    if init_pop:
        pop = [copy.deepcopy(g) for g in init_pop]
    while len(pop) < config.pop_size:
        g = mutate_genome(
            DEFAULT_GENOME,
            MutateParams(sigma_frac=0.25, per_gene_prob=0.9, reset_prob=0.25,
                         seed=rng.randint(0, 10**9))
        )
        pop.append(g)

    # evaluate entire population together
    fits, ctrl_fits = evaluate_population_all_in_one(
        pop_genomes=pop,
        games_per_eval=config.games_per_eval,
        base_seed=config.eval_seed,
        verbose_game=0
    )
    scored: List[Tuple[Dict[str, Any], Fitness]] = list(zip(pop, fits))

    # track best
    best_g, best_f = scored[0]
    for g, f in scored[1:]:
        if lexicographic_better(f, best_f):
            best_g, best_f = copy.deepcopy(g), f

    for gen in range(config.generations):
        # elites
        elites = sorted(scored, key=lambda gf: (
            gf[1].median, gf[1].win_rate, gf[1].max_score, gf[1].min_score
        ), reverse=True)[:config.elitism]

        # breed rest
        new_pop: List[Dict[str, Any]] = [copy.deepcopy(g) for g, _ in elites]
        while len(new_pop) < config.pop_size:
            parents = tournament_select(scored, tourney_size=config.tourney_size, k=2, rng=rng)
            child = crossover(parents[0], parents[1], config.crossover_params)
            if config.mutation_after_crossover:
                child = mutate_genome(child, config.mutate_params)
            new_pop.append(child)

        # replace pop and re-evaluate everyone together
        pop = new_pop
        fits, ctrl_fits = evaluate_population_all_in_one(
            pop_genomes=pop,
            games_per_eval=config.games_per_eval,
            base_seed=config.eval_seed + gen + 1,  # rotate seed per gen
            verbose_game=0
        )
        scored = list(zip(pop, fits))

        # best-of
        gen_best_g, gen_best_f = scored[0]
        for g, f in scored[1:]:
            if lexicographic_better(f, gen_best_f):
                gen_best_g, gen_best_f = g, f
        if lexicographic_better(gen_best_f, best_f):
            best_g, best_f = copy.deepcopy(gen_best_g), gen_best_f

        if (gen + 1) % config.log_every == 0:
            print(f"[Gen {gen+1}/{config.generations}] "
                  f"best median={best_f.median:.2f} win%={100*best_f.win_rate:.2f} "
                  f"max={best_f.max_score} min={best_f.min_score} "
                  f"(bust%={100*best_f.diagnostics.get('bust_rate', 0):.2f}, "
                  f"games={config.games_per_eval}, pop={config.pop_size})")
            _print_controls_summary(ctrl_fits)

    return best_g, best_f


# =========================
# Quick CLI
# =========================
if __name__ == "__main__":
    cfg = GARunConfig(
        pop_size=50,             # everyone joins the table
        generations=10,
        games_per_eval=500,      # increase later as needed
        elitism=5,
        tourney_size=4,
        mutation_after_crossover=True,
        mutate_params=MutateParams(sigma_frac=0.12, per_gene_prob=0.35, reset_prob=0.06),
        crossover_params=CrossoverParams(alpha_min=0.3, alpha_max=0.7, per_gene_prob=0.9),
        eval_seed=42,
        log_every=1,
    )
    best_g, best_f = evolve(cfg)
    print("\n=== Best genome ===")
    print(json.dumps(best_g, indent=2))
    print("\n=== Best fitness ===")
    print(f"median={best_f.median:.2f}, win%={100*best_f.win_rate:.2f}, "
          f"max={best_f.max_score}, min={best_f.min_score}, "
          f"bust%={100*best_f.diagnostics.get('bust_rate', 0):.2f})")
