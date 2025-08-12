# ga_evaluator.py
import random, statistics as stats
from typing import Any, Dict, List, Tuple
from .ga_types import Fitness
from .ga_genome import Genome
from .ga_controls import ControlPool
from evo_player import EvoPlayer
from game import SuitsGambitGame

class PopulationEvaluator:
    """
    Fitness = WIN RATE ONLY.
    We still run normal games, but the returned Fitness objects are scored purely by
    win_rate. All other numbers are provided only as diagnostics for logging.
    """
    def __init__(self, controls: ControlPool):
        self.controls = controls

    def evaluate_all_in_one(self,
                            pop: List[Genome],
                            games_per_eval: int = 1000,
                            base_seed: int = 123,
                            verbose_game: int = 0
                            ) -> Tuple[List[Fitness], Dict[str, Fitness]]:
        P = len(pop)
        ctrl_players = self.controls.make()
        ctrl_names = [c.name for c in ctrl_players]

        # --- Accumulators (minimized; we only *need* wins) ---
        wins = [0 for _ in range(P)]
        ctrl_wins: Dict[str, int] = {n: 0 for n in ctrl_names}

        # Kept as diagnostics only (not used for fitness now)
        totals = [[] for _ in range(P)]
        max_score = [0 for _ in range(P)]
        min_score = [None for _ in range(P)]
        bust_rounds = [0 for _ in range(P)]
        rounds_played = [0 for _ in range(P)]

        ctrl_totals: Dict[str, List[int]] = {n: [] for n in ctrl_names}
        ctrl_max: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_min: Dict[str, int | None] = {n: None for n in ctrl_names}
        ctrl_bust_rounds: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_rounds_played: Dict[str, int] = {n: 0 for n in ctrl_names}

        # --- Simulate ---
        for gidx in range(games_per_eval):
            # Fresh Evo players each game
            evos = [EvoPlayer(f"Evo{i}", genome=pop[i].to_json()) for i in range(P)]
            # Mix with controls and shuffle seating deterministically per gidx
            players = evos + self.controls.make()
            random.Random(base_seed + gidx).shuffle(players)

            game = SuitsGambitGame(players, verbose=verbose_game, seed=(base_seed + gidx))
            winner, results = game.play()

            name_to_idx = {f"Evo{i}": i for i in range(P)}
            for p in players:
                score = results[p.name]

                if p.name in name_to_idx:
                    i = name_to_idx[p.name]
                    # diagnostics only
                    totals[i].append(score)
                    max_score[i] = max(max_score[i], score)
                    min_val = min_score[i]
                    min_score[i] = score if min_val is None else min(min_val, score)
                    rounds_played[i] += len(p.round_scores)
                    bust_rounds[i] += sum(1 for s in p.round_scores if s == 0)

                elif p.name in ctrl_totals:
                    ctrl_totals[p.name].append(score)
                    ctrl_max[p.name] = max(ctrl_max[p.name], score)
                    cmin = ctrl_min[p.name]
                    ctrl_min[p.name] = score if cmin is None else min(cmin, score)
                    ctrl_rounds_played[p.name] += len(p.round_scores)
                    ctrl_bust_rounds[p.name] += sum(1 for s in p.round_scores if s == 0)

            # WIN RATE ONLY: count wins; ties give no credit
            if winner in name_to_idx:
                wins[name_to_idx[winner]] += 1
            elif winner in ctrl_wins:
                ctrl_wins[winner] += 1

        # --- Finalize genomes (fitness = win_rate only) ---
        fits: List[Fitness] = []
        for i in range(P):
            wr = wins[i] / max(1, games_per_eval)

            # Everything else is diagnostics only
            n = len(totals[i])
            mean = (sum(totals[i]) / n) if n else 0.0
            sd = (stats.pstdev(totals[i]) if n > 1 else 0.0)
            med = float(stats.median(totals[i])) if n else 0.0
            mx = int(max_score[i]) if n else 0
            mn = int(min_score[i]) if (min_score[i] is not None) else 0
            br = (bust_rounds[i] / rounds_played[i]) if rounds_played[i] else 0.0

            fits.append(Fitness(
                # Only win_rate matters; set these neutrally if your GA ignores them,
                # or leave median for logging but DO NOT sort by it.
                median=0.0,               # neutral placeholder
                win_rate=wr,              # <<< FITNESS DIMENSION
                max_score=0,              # neutral
                min_score=0,              # neutral
                diagnostics={
                    "median_observed": med,
                    "mean": mean,
                    "sd": sd,
                    "max": mx,
                    "min": mn,
                    "bust_rate": br,
                    "n_games": n,
                }
            ))

        # --- Finalize controls (win_rate only, rest diagnostics) ---
        control_fits: Dict[str, Fitness] = {}
        for n, ts in ctrl_totals.items():
            wr = ctrl_wins[n] / max(1, games_per_eval)
            count = len(ts)
            mean = (sum(ts) / count) if count else 0.0
            sd = (stats.pstdev(ts) if count > 1 else 0.0)
            med = float(stats.median(ts)) if count else 0.0
            mx = int(ctrl_max[n]) if count else 0
            mn = int(ctrl_min[n]) if (ctrl_min[n] is not None) else 0
            rounds = ctrl_rounds_played[n]
            br = (ctrl_bust_rounds[n] / rounds) if rounds else 0.0

            control_fits[n] = Fitness(
                median=0.0,              # neutral
                win_rate=wr,             # <<< FITNESS DIMENSION
                max_score=0,             # neutral
                min_score=0,             # neutral
                diagnostics={
                    "median_observed": med,
                    "mean": mean,
                    "sd": sd,
                    "max": mx,
                    "min": mn,
                    "bust_rate": br,
                    "n_games": count,
                }
            )

        return fits, control_fits
