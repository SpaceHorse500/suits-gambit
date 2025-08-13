# suits_gambit_game.py
from typing import List, Dict, Any, Optional, Tuple
from cards import Deck
from players import BasePlayer
from utils import evaluate_expression, expr_string_annotated

ROUNDS = 5

class SuitsGambitGame:
    def __init__(self, players: List[BasePlayer], verbose: int | bool = 0,
                 seed: Optional[int] = None, stats: Optional["StatsCollector"] = None):
        assert len(players) >= 2, "Need at least two players."
        self.players = players
        self.verbose = 2 if isinstance(verbose, bool) and verbose else int(verbose)
        if seed is not None:
            import random
            random.seed(seed)
        self.stats = stats
        self._log = self._setup_logger()

    def _setup_logger(self):
        if self.verbose >= 2:
            return lambda msg: print(msg)
        return lambda msg: None

    def _public_state(self) -> Dict[str, Any]:
        return {
            "players_public": [
                {"name": p.name, "scores": list(p.round_scores), "ops": list(p.ops_between)}
                for p in self.players
            ],
            "rounds": ROUNDS,
            "turn_order": [p.name for p in self.players],
        }

    def _play_single_turn(self, player: BasePlayer, round_idx: int, public_state: Dict[str, Any]) -> Tuple[int, Optional[int]]:
        deck = Deck()
        deck.reset()
        points = 0
        pre_bust: Optional[int] = None
        last_card = None
        last_forbidden_suit: Optional[str] = None  # FIX: explicit init

        while True:
            # Draw new card and declare new forbidden suit
            card = deck.draw()
            if not card:
                self._log(f"  [{player.name}] deck exhausted -> auto-stop with {points}")
                break

            self._log(f"    draw: {card.rank}{card.suit} (rem {deck.remaining()})")

            forbidden_ctx = {
                "phase": "declare_forbidden",
                "round_index": round_idx,
                "public": public_state,
                "info_card": {"rank": card.rank, "suit": card.suit},
                "deck_remaining_by_suit": deck.remaining_by_suit(),
                "last_card": last_card,
                "current_points": points,
            }
            forbidden_suit = player.choose_forbidden_suit(card, forbidden_ctx)
            self._log(f"    [{player.name}] new forbidden suit: {forbidden_suit}")

            # Bust check uses the PREVIOUS forbidden suit (applies to this draw)
            if last_card and last_forbidden_suit is not None and card.suit == last_forbidden_suit:
                pre_bust = points
                self._log(f"    -> BUST! [{player.name}] scores 0 this round.")
                points = 0
                break

            points += 1
            last_card = card
            last_forbidden_suit = forbidden_suit

            # Decision to continue/stop (after at least 1 point)
            if points >= 1:
                decision_ctx = {
                    "phase": "draw_decision",
                    "round_index": round_idx,
                    "public": public_state,
                    "current_points": points,
                    "last_card": {"rank": card.rank, "suit": card.suit},
                    "deck_remaining_by_suit": deck.remaining_by_suit(),
                    "current_forbidden": forbidden_suit,  # FIX: explicit for p_bust
                }
                action = player.choose_continue_or_stop(points, decision_ctx)
                if action == "stop":
                    break

        self._log(f"  [{player.name}] ends Round {round_idx} with: {points}\n")
        return points, pre_bust

    def _operator_pick_phase(self, round_idx: int, public_state: Dict[str, Any]):
        self._log(f"Operator picks between R{round_idx} & R{round_idx+1}:")
        previous_picks = []
        all_scores = {p.name: list(p.round_scores) for p in self.players}

        for p in self.players:
            ctx = {
                "phase": "operator_pick",
                "round_index": round_idx,
                "public": public_state,
                "previous_picks": list(previous_picks),
                "picker": p.name,
            }
            op = p.choose_operator_between_rounds(
                my_scores=list(p.round_scores),
                all_scores=all_scores,
                previous_picks=list(previous_picks),
                ctx=ctx,
            )
            # FIX: normalize operators — accept 'x', '×', '*'
            if op in ("x", "×", "*"):
                op = "x"
            else:
                op = "+"
            p.ops_between.append(op)
            previous_picks.append({"player": p.name, "op": op})
            self._log(f"  -> {p.name}: {op}")

    def play(self):
        for p in self.players:
            p.reset()
            setattr(p, "pre_bust", [])

        for r in range(1, ROUNDS + 1):
            public_state = self._public_state()
            self._log(f"\n=== Round {r} ===")

            for p in self.players:
                score, pre_bust = self._play_single_turn(p, r, public_state)
                p.round_scores.append(score)
                p.pre_bust.append(pre_bust)

                if self.stats and hasattr(self.stats, "record_round"):
                    op_ctx = p.ops_between[r - 2] if r > 1 and len(p.ops_between) >= (r - 1) else "+"
                    outcome = RoundOutcome(  # type: ignore[name-defined]
                        player=p.name,
                        round_idx=r,
                        points=score,
                        busted=(score == 0),
                        pre_bust_streak=pre_bust,
                        op_context=op_ctx
                    )
                    self.stats.record_round(outcome)

            if r < ROUNDS:
                self._operator_pick_phase(r, public_state)

        results = {p.name: evaluate_expression(p.round_scores, p.ops_between) for p in self.players}

        if self.verbose >= 1:
            print("\n=== FINAL ===")
            for p in self.players:
                line = expr_string_annotated(p.round_scores, p.ops_between, p.pre_bust)
                print(f"{p.name}: {line} = {results[p.name]}")

        best = max(results.values()) if results else None
        leaders = [n for n, v in results.items() if v == best] if best is not None else []
        winner = leaders[0] if len(leaders) == 1 else None

        return winner, results
