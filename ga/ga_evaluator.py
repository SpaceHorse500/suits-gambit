# ga/ga_evaluator.py
import math
import random
import statistics as stats
from typing import Any, Dict, List, Tuple

from .ga_types import Fitness
from .ga_genome import Genome
from .ga_controls import ControlPool
from evo_player import EvoPlayer
from game import SuitsGambitGame


def _quartiles(values: List[int]) -> tuple[float, float]:
    """Return (Q1, Q3). Robust for small n; falls back to linear percentile."""
    if not values:
        return 0.0, 0.0
    try:
        qs = stats.quantiles(values, n=4, method="inclusive")  # [Q1, median, Q3]
        return float(qs[0]), float(qs[2])
    except Exception:
        s = sorted(values)
        n = len(s)
        if n == 1:
            return float(s[0]), float(s[0])

        def pctl(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return float(s[f])
            return float(s[f] + (s[c] - s[f]) * (k - f))

        return pctl(0.25), pctl(0.75)


def _avg_tie_ranks(name_to_score: Dict[str, int]) -> Dict[str, float]:
    """
    Average-of-ties ranking: best score gets rank 1.0; ties share the mean rank.
    Returns map: name -> rank (float).
    """
    items = sorted(name_to_score.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: Dict[str, float] = {}
    pos = 1
    i = 0
    n = len(items)
    while i < n:
        j = i
        score_i = items[i][1]
        while j < n and items[j][1] == score_i:
            j += 1
        # items[i:j] share the same score
        k = j - i
        # average rank of positions [pos, pos+k-1]
        avg_rank = (pos + (pos + k - 1)) / 2.0
        for t in range(i, j):
            ranks[items[t][0]] = avg_rank
        pos += k
        i = j
    return ranks


class PopulationEvaluator:
    """
    Fitness = multi-criteria:
      1) win_rate (primary),
      2) median score (secondary),
      3) Q1, then Q3 (tertiary).
    Diagnostics include: sd/iqr, bust_rate, rank_avg, rank_pct_avg, rank_hist (10 deciles).
    """
    def __init__(self, controls: ControlPool):
        self.controls = controls

    def evaluate_all_in_one(
        self,
        pop: List[Genome],
        games_per_eval: int = 1000,
        base_seed: int = 123,
        verbose_game: int = 0
    ) -> Tuple[List[Fitness], Dict[str, Fitness]]:

        P = len(pop)
        ctrl_players = self.controls.make()
        ctrl_names = [c.name for c in ctrl_players]

        # --- Accumulators (genomes) ---
        wins = [0 for _ in range(P)]
        totals: List[List[int]] = [[] for _ in range(P)]
        max_score = [0 for _ in range(P)]
        min_score = [None for _ in range(P)]
        bust_rounds = [0 for _ in range(P)]
        rounds_played = [0 for _ in range(P)]
        # ranking accumulators
        rank_sum = [0.0 for _ in range(P)]
        rank_count = [0 for _ in range(P)]
        rank_hist = [[0 for _ in range(10)] for _ in range(P)]  # deciles of percentile rank

        # --- Accumulators (controls) ---
        ctrl_wins: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_totals: Dict[str, List[int]] = {n: [] for n in ctrl_names}
        ctrl_max: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_min: Dict[str, int | None] = {n: None for n in ctrl_names}
        ctrl_bust_rounds: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_rounds_played: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_rank_sum: Dict[str, float] = {n: 0.0 for n in ctrl_names}
        ctrl_rank_count: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_rank_hist: Dict[str, List[int]] = {n: [0]*10 for n in ctrl_names}

        # --- Simulate ---
        for gidx in range(games_per_eval):
            # Fresh Evo players each game
            evos = [EvoPlayer(f"Evo{i}", genome=pop[i].to_json()) for i in range(P)]
            # Mix with controls and shuffle seating deterministically per gidx
            players = evos + self.controls.make()
            rnd = random.Random(base_seed + gidx)
            rnd.shuffle(players)

            game = SuitsGambitGame(players, verbose=verbose_game, seed=(base_seed + gidx))
            winner, results = game.play()

            # build reverse lookup for evos
            name_to_idx = {f"Evo{i}": i for i in range(P)}

            # --- ranking for this game (all players together) ---
            per_game_ranks = _avg_tie_ranks(results)
            M = len(players)
            denom = max(1, M - 1)  # for percentile normalization

            # --- per-player accounting ---
            for p in players:
                score = results[p.name]
                rnk = per_game_ranks[p.name]
                pct = (rnk - 1.0) / denom  # 0.0 best .. 1.0 worst
                bin_idx = min(9, int(pct * 10.0))  # 0..9

                if p.name in name_to_idx:  # evo
                    i = name_to_idx[p.name]
                    totals[i].append(score)
                    max_score[i] = max(max_score[i], score)
                    mn = min_score[i]
                    min_score[i] = score if mn is None else min(mn, score)
                    rounds_played[i] += len(p.round_scores)
                    bust_rounds[i] += sum(1 for s in p.round_scores if s == 0)
                    # rank
                    rank_sum[i] += rnk
                    rank_count[i] += 1
                    rank_hist[i][bin_idx] += 1
                else:  # control
                    ctrl_totals[p.name].append(score)
                    ctrl_max[p.name] = max(ctrl_max[p.name], score)
                    cmn = ctrl_min[p.name]
                    ctrl_min[p.name] = score if cmn is None else min(cmn, score)
                    ctrl_rounds_played[p.name] += len(p.round_scores)
                    ctrl_bust_rounds[p.name] += sum(1 for s in p.round_scores if s == 0)
                    # rank
                    ctrl_rank_sum[p.name] += rnk
                    ctrl_rank_count[p.name] += 1
                    ctrl_rank_hist[p.name][bin_idx] += 1

            # WIN RATE only counts outright wins; ties give no credit
            if winner in name_to_idx:
                wins[name_to_idx[winner]] += 1
            elif winner in ctrl_wins:
                ctrl_wins[winner] += 1

        # --- Finalize genomes ---
        fits: List[Fitness] = []
        for i in range(P):
            wr = wins[i] / max(1, games_per_eval)

            n = len(totals[i])
            med = float(stats.median(totals[i])) if n else 0.0
            q1, q3 = _quartiles(totals[i]) if n else (0.0, 0.0)
            mean = (sum(totals[i]) / n) if n else 0.0
            sd = stats.pstdev(totals[i]) if n > 1 else 0.0
            mx = int(max_score[i]) if n else 0
            mn = int(min_score[i]) if (min_score[i] is not None) else 0
            br = (bust_rounds[i] / rounds_played[i]) if rounds_played[i] else 0.0
            iqr = q3 - q1

            rcount = max(1, rank_count[i])
            ravg = rank_sum[i] / rcount
            # convert avg rank to avg percentile for convenience
            M = P + len(ctrl_names)
            denom = max(1, M - 1)
            r_pct_avg = (ravg - 1.0) / denom
            rhist = list(rank_hist[i])

            fits.append(Fitness(
                win_rate=wr,
                median=med,
                q1=q1,
                q3=q3,
                max_score=mx,
                min_score=mn,
                diagnostics={
                    "mean": mean,
                    "sd": sd,
                    "iqr": iqr,
                    "bust_rate": br,
                    "n_games": n,
                    "rank_avg": ravg,            # 1.0 is best
                    "rank_pct_avg": r_pct_avg,   # 0.0 best .. 1.0 worst
                    "rank_hist": rhist,          # 10 deciles
                }
            ))

        # --- Finalize controls (for logging) ---
        control_fits: Dict[str, Fitness] = {}
        for nme, ts in ctrl_totals.items():
            wr = ctrl_wins[nme] / max(1, games_per_eval)
            count = len(ts)
            med = float(stats.median(ts)) if count else 0.0
            q1, q3 = _quartiles(ts) if count else (0.0, 0.0)
            mean = (sum(ts) / count) if count else 0.0
            sd = stats.pstdev(ts) if count > 1 else 0.0
            mx = int(ctrl_max[nme]) if count else 0
            mn = int(ctrl_min[nme]) if (ctrl_min[nme] is not None) else 0
            rounds = ctrl_rounds_played[nme]
            br = (ctrl_bust_rounds[nme] / rounds) if rounds else 0.0
            iqr = q3 - q1

            rcount = max(1, ctrl_rank_count[nme])
            ravg = ctrl_rank_sum[nme] / rcount
            M = P + len(ctrl_names)
            denom = max(1, M - 1)
            r_pct_avg = (ravg - 1.0) / denom
            rhist = list(ctrl_rank_hist[nme])

            control_fits[nme] = Fitness(
                win_rate=wr,
                median=med,
                q1=q1,
                q3=q3,
                max_score=mx,
                min_score=mn,
                diagnostics={
                    "mean": mean,
                    "sd": sd,
                    "iqr": iqr,
                    "bust_rate": br,
                    "n_games": count,
                    "rank_avg": ravg,
                    "rank_pct_avg": r_pct_avg,
                    "rank_hist": rhist,
                }
            )

        return fits, control_fits
