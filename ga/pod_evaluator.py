# ga/pod_evaluator.py
from __future__ import annotations
import math, random, statistics as stats
from typing import Any, Dict, List, Tuple, Callable, Iterable

from evo_player import EvoPlayer
from game import SuitsGambitGame

# Fallback: we use TacticianPlayer as the "Meta" control.
try:
    from tactician_player import TacticianPlayer as MetaPlayer
except Exception:
    from smart_player import SmartPlayer as MetaPlayer  # graceful fallback

# Reuse your Fitness dataclass to avoid breaking code
try:
    from .ga_types import Fitness
except Exception:
    from dataclasses import dataclass
    @dataclass
    class Fitness:
        median: float
        win_rate: float
        max_score: int
        min_score: int
        diagnostics: Dict[str, Any]


def pairwise_elo_update(
    ratings: Dict[str, float],
    names: List[str],
    ranks: Dict[str, float],
    k_factor: float = 24.0,
) -> None:
    """
    Multi-player Elo via pairwise comparisons:
    - If A ranks better than B -> A gets a 'win' vs B (1.0), else 0.0; ties => 0.5
    - One pass, in-place updates on 'ratings'
    """
    # Pre-compute expected scores
    expected: Dict[Tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ra, rb = ratings[a], ratings[b]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            expected[(a, b)] = ea

    # Accumulate deltas then apply (so order doesn't bias)
    delta: Dict[str, float] = {n: 0.0 for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            # Rank: lower number = better (1st place best)
            if abs(ranks[a] - ranks[b]) < 1e-9:
                sa = 0.5
            elif ranks[a] < ranks[b]:
                sa = 1.0
            else:
                sa = 0.0
            ea = expected[(a, b)]
            eb = 1.0 - ea

            delta[a] += k_factor * (sa - ea)
            delta[b] += k_factor * ((1.0 - sa) - eb)

    for n in names:
        ratings[n] += delta[n]


def rank_from_scores(scores: Dict[str, int]) -> Dict[str, float]:
    """
    Returns 1-based ranks with ties averaged (competition ranking -> average rank).
    Higher score is better.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and items[j][1] == items[i][1]:
            j += 1
        # items[i:j] tie block occupies places i..j-1 (0-based)
        # their rank is the average of 1-based positions
        start_place = i + 1
        end_place = j
        avg_rank = (start_place + end_place) / 2.0
        for k in range(i, j):
            ranks[items[k][0]] = avg_rank
        i = j
    return ranks


def borda_points(rank: float, pod_size: int) -> float:
    """
    Linear rank points in [0,1]: 1st -> 1.0; last -> 0.0.
    Ties handled by passing averaged rank.
    """
    return (pod_size - rank) / (pod_size - 1)


class PodEloEvaluator:
    """
    Evaluate genomes in pods of fixed size (default 5):
      - Each pod = (pod_size - 1) Evo genomes + 1 Meta control
      - Seat rotation each round
      - Pairwise Elo updates per game
      - Returns Fitness list (per genome) + control Fitness

    Notes:
      * Runtime is dominated by game simulation; Elo math is negligible.
      * If genomes don't divide evenly into (pod_size-1), the last pod
        is padded with extra Meta controls (so every game has exactly pod_size players).
    """

    def __init__(
        self,
        pod_size: int = 5,
        control_factory: Callable[[str], Any] | None = None,
        initial_elo: float = 1500.0,
        k_factor: float = 24.0,
    ):
        assert pod_size >= 3, "pod_size must be at least 3"
        self.pod_size = pod_size
        self.gpp = pod_size - 1  # genomes per pod (1 seat reserved for Meta)
        self.initial_elo = initial_elo
        self.k = k_factor
        self.control_factory = control_factory or (lambda name: MetaPlayer(name))

    def _make_pods(self, indices: List[int], rng: random.Random) -> List[List[int]]:
        """Chunk genome indices into pods of size self.gpp (4 when pod_size=5)."""
        pods: List[List[int]] = []
        for i in range(0, len(indices), self.gpp):
            pods.append(indices[i : i + self.gpp])
        return pods

    def evaluate(
        self,
        pop: List[Any],                  # Genome objects (must have .to_json())
        rounds: int = 100,              # how many scheduling rounds
        base_seed: int = 12345,
        verbose_game: int = 0,          # pass to SuitsGambitGame
    ) -> Tuple[List[Fitness], Dict[str, Fitness]]:
        P = len(pop)
        rng_master = random.Random(base_seed)

        # Ratings and accumulators
        elo: Dict[str, float] = {f"Evo{i}": self.initial_elo for i in range(P)}
        elo_meta: float = self.initial_elo  # shared Meta rating

        totals: List[List[int]] = [[] for _ in range(P)]
        wins = [0 for _ in range(P)]
        top2 = [0 for _ in range(P)]
        ranks_accum: List[List[float]] = [[] for _ in range(P)]
        rank_pts_sum = [0.0 for _ in range(P)]
        bust_rounds = [0 for _ in range(P)]
        rounds_played = [0 for _ in range(P)]
        max_score = [0 for _ in range(P)]
        min_score = [None for _ in range(P)]

        # Control accumulators (Meta across all pods)
        meta_totals: List[int] = []
        meta_wins = 0
        meta_bust_rounds = 0
        meta_rounds_played = 0
        meta_max = 0
        meta_min: int | None = None

        # Scheduling
        for r in range(rounds):
            # Shuffle genome indices; make pods of gpp
            idxs = list(range(P))
            rng_round = random.Random(rng_master.randint(0, 2**31 - 1))
            rng_round.shuffle(idxs)
            pods = self._make_pods(idxs, rng_round)

            for p_idx, pod in enumerate(pods):
                # Build player list for this pod
                players = []
                evo_names: List[str] = []
                for gi in pod:
                    name = f"Evo{gi}"
                    players.append(EvoPlayer(name, genome=pop[gi].to_json()))
                    evo_names.append(name)

                # If pod has fewer than gpp genomes, pad with extra Metas
                while len(players) < self.gpp:
                    players.append(self.control_factory(f"Meta_pad{p_idx}_{len(players)}"))

                # Always one Meta seat
                players.append(self.control_factory(f"Meta_{r}_{p_idx}"))

                # Seat rotation: rotate by (r + p_idx) to balance seats across rounds/pods
                rot = (r + p_idx) % self.pod_size
                players = players[rot:] + players[:rot]

                # Run the game
                seed = base_seed ^ (r * 1315423911 + p_idx * 2654435761)
                game = SuitsGambitGame(players, verbose=verbose_game, seed=seed)
                winner, results = game.play()  # results: name -> total

                # Compute ranks (+ rank points)
                ranks = rank_from_scores(results)
                rpts = {n: borda_points(ranks[n], self.pod_size) for n in ranks}

                # Elo: include Meta as one identity (shared rating)
                names_this_game = [p.name for p in players]
                # Temporary ratings view for this game
                tmp_ratings: Dict[str, float] = {}
                for n in names_this_game:
                    if n.startswith("Evo"):
                        tmp_ratings[n] = elo[n]
                    else:
                        tmp_ratings[n] = elo_meta

                pairwise_elo_update(tmp_ratings, names_this_game, ranks, k_factor=self.k)

                # Write back updated ratings
                for n in names_this_game:
                    if n.startswith("Evo"):
                        elo[n] = tmp_ratings[n]
                    else:
                        # Merge all Meta clones into single shared elo
                        elo_meta = tmp_ratings[n]

                # Collect stats
                name_to_idx = {f"Evo{i}": i for i in range(P)}
                # Determine first/second place sets for quick win/top2
                best_score = max(results.values())
                # ranks: lower is better; 1.0 is 1st
                second_rank = sorted(set(ranks.values()))[1] if len(set(ranks.values())) > 1 else 1.0

                for pl in players:
                    total = results[pl.name]
                    is_meta = not pl.name.startswith("Evo")
                    if is_meta:
                        meta_totals.append(total)
                        meta_max = max(meta_max, total)
                        meta_min = total if meta_min is None else min(meta_min, total)
                        meta_rounds_played += len(pl.round_scores)
                        meta_bust_rounds += sum(1 for s in pl.round_scores if s == 0)
                    else:
                        i = name_to_idx[pl.name]
                        totals[i].append(total)
                        max_score[i] = max(max_score[i], total)
                        min_score[i] = total if min_score[i] is None else min(min_score[i], total)
                        rounds_played[i] += len(pl.round_scores)
                        bust_rounds[i] += sum(1 for s in pl.round_scores if s == 0)
                        ranks_accum[i].append(ranks[pl.name])
                        rank_pts_sum[i] += rpts[pl.name]

                # Wins / Top2
                winners = [n for n, s in results.items() if s == best_score]
                seconders = [n for n, rk in ranks.items() if abs(rk - second_rank) < 1e-9]
                for n in winners:
                    if n.startswith("Evo"):
                        wins[name_to_idx[n]] += 1
                    else:
                        meta_wins += 1
                for n in seconders:
                    if n.startswith("Evo"):
                        top2[name_to_idx[n]] += 1

        # Prepare Fitness outputs (per genome)
        fits: List[Fitness] = []
        baseline_win = 1.0 / self.pod_size  # random baseline in 5-player pod = 0.2
        for i in range(P):
            ts = totals[i]
            if not ts:
                fits.append(Fitness(0.0, 0.0, 0, 0,
                    {"elo": self.initial_elo, "mean": 0.0, "sd": 0.0, "bust_rate": 1.0,
                     "rank_mean": float('inf'), "rank_points": 0.0, "lift_vs_baseline": 0.0, "n_games": 0}))
                continue

            med = float(stats.median(ts))
            mx = int(max_score[i])
            mn = int(min_score[i]) if min_score[i] is not None else 0
            wr = wins[i] / len(ts)
            br = (bust_rounds[i] / rounds_played[i]) if rounds_played[i] else 1.0
            rank_mean = sum(ranks_accum[i]) / len(ranks_accum[i]) if ranks_accum[i] else float('inf')
            rank_pts_mean = rank_pts_sum[i] / len(ts) if ts else 0.0
            mean_total = sum(ts) / len(ts)
            sd_total = stats.pstdev(ts) if len(ts) > 1 else 0.0
            lift = (wr / baseline_win) if baseline_win > 0 else 0.0
            fits.append(Fitness(
                median=med,
                win_rate=wr,
                max_score=mx,
                min_score=mn,
                diagnostics={
                    "elo": elo[f"Evo{i}"],
                    "mean": mean_total,
                    "sd": sd_total,
                    "bust_rate": br,
                    "rank_mean": rank_mean,
                    "rank_points": rank_pts_mean,
                    "top2_rate": top2[i] / len(ts),
                    "lift_vs_baseline": lift,
                    "n_games": len(ts),
                }
            ))

        # Control Fitness (Meta across all pods)
        if meta_totals:
            meta_wr = meta_wins / len(meta_totals)
            meta_med = float(stats.median(meta_totals))
            meta_mean = sum(meta_totals) / len(meta_totals)
            meta_sd = stats.pstdev(meta_totals) if len(meta_totals) > 1 else 0.0
            meta_br = (meta_bust_rounds / meta_rounds_played) if meta_rounds_played else 1.0
            control_fits = {
                "Meta": Fitness(
                    median=meta_med,
                    win_rate=meta_wr,
                    max_score=int(meta_max),
                    min_score=int(meta_min) if meta_min is not None else 0,
                    diagnostics={
                        "elo": elo_meta,
                        "mean": meta_mean,
                        "sd": meta_sd,
                        "bust_rate": meta_br,
                        "n_games": len(meta_totals),
                    },
                )
            }
        else:
            control_fits = {
                "Meta": Fitness(0.0, 0.0, 0, 0,
                    {"elo": self.initial_elo, "mean": 0.0, "sd": 0.0, "bust_rate": 1.0, "n_games": 0})
            }

        return fits, control_fits
