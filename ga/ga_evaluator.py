# ga_evaluator.py
import random, statistics as stats
from typing import Any, Dict, List, Tuple
from .ga_types import Fitness
from .ga_genome import Genome
from .ga_controls import ControlPool
from evo_player import EvoPlayer
from game import SuitsGambitGame

class PopulationEvaluator:
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

        # accumulators for genomes
        totals = [[] for _ in range(P)]
        wins = [0 for _ in range(P)]
        bust_rounds = [0 for _ in range(P)]
        rounds_played = [0 for _ in range(P)]
        max_score = [0 for _ in range(P)]
        min_score = [None for _ in range(P)]

        # accumulators for controls
        ctrl_totals: Dict[str, List[int]] = {n: [] for n in ctrl_names}
        ctrl_wins: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_bust_rounds: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_rounds_played: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_max: Dict[str, int] = {n: 0 for n in ctrl_names}
        ctrl_min: Dict[str, int | None] = {n: None for n in ctrl_names}

        for gidx in range(games_per_eval):
            evos = [EvoPlayer(f"Evo{i}", genome=pop[i].to_json()) for i in range(P)]
            players = evos + self.controls.make()
            random.Random(base_seed + gidx).shuffle(players)

            game = SuitsGambitGame(players, verbose=verbose_game, seed=(base_seed + gidx))
            winner, results = game.play()

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
                median=med, win_rate=wr, max_score=mx, min_score=mn,
                diagnostics={"bust_rate": br, "mean": sum(ts)/len(ts), "sd": (stats.pstdev(ts) if len(ts)>1 else 0.0), "n_games": len(ts)}
            ))

        # finalize controls
        control_fits: Dict[str, Fitness] = {}
        for n, ts in ctrl_totals.items():
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
                median=med, win_rate=wr, max_score=mx, min_score=mn,
                diagnostics={"bust_rate": br, "mean": sum(ts)/len(ts), "sd": (stats.pstdev(ts) if len(ts)>1 else 0.0), "n_games": len(ts)}
            )
        return fits, control_fits
