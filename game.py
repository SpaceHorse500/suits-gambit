# game.py
from typing import List, Dict, Any, Optional, Tuple
from cards import Deck
from players import BasePlayer
from utils import evaluate_expression, expr_string_annotated

ROUNDS = 5

# Optional import of extra stats
try:
    from stats_extra import StatsCollector, RoundOutcome
except ImportError:
    StatsCollector = None
    RoundOutcome = None


class SuitsGambitGame:
    """
    Turn-based Suits Gambit supporting N players.
    
    Verbosity levels:
      - 0: no printing
      - 1: only the final concise scoreboard (with [k].0 bust annotations)
      - 2: full detailed logs (per-draw, picks, etc.)
    """
    def __init__(self, players: List[BasePlayer], verbose: int | bool = 0,
                 seed: Optional[int] = None, stats: Optional["StatsCollector"] = None):
        assert len(players) >= 2, "Need at least two players."
        self.players = players
        if isinstance(verbose, bool):
            self.verbose = 2 if verbose else 0
        else:
            self.verbose = int(verbose)
        if seed is not None:
            import random
            random.seed(seed)
        self.stats = stats
        self._log = self._setup_logger()

    def _setup_logger(self):
        """Pre-configure the log function to avoid repeated checks."""
        if self.verbose >= 2:
            def log_func(msg: str):
                print(msg)
            return log_func
        else:
            def log_func(msg: str):
                pass
            return log_func

    def _public_state(self) -> Dict[str, Any]:
        """Generates the public state, now cached and reused."""
        return {
            "players_public": [
                {"name": p.name, "scores": list(p.round_scores), "ops": list(p.ops_between)}
                for p in self.players
            ],
            "rounds": ROUNDS,
            "turn_order": [p.name for p in self.players],
        }

    def _play_single_turn(self, player: BasePlayer, round_idx: int, public_state: Dict[str, Any]) -> Tuple[int, Optional[int]]:
        """
        One player's turn with a fresh deck.
        Returns (score, pre_bust_streak) where pre_bust_streak is None if no bust occurred.
        """
        deck = Deck()
        deck.reset()

        first = deck.draw()
        assert first is not None
        self._log(f"  [{player.name}] info card: {first.rank}{first.suit} (rem {deck.remaining()})")

        # Declare forbidden suit
        forbidden_ctx: Dict[str, Any] = {
            "phase": "declare_forbidden",
            "round_index": round_idx,
            "public": public_state,
            "info_card": {"rank": first.rank, "suit": first.suit},
            "deck_remaining_by_suit": deck.remaining_by_suit(),
        }
        forbidden = player.choose_forbidden_suit(first, forbidden_ctx)
        self._log(f"  [{player.name}] forbidden suit: {forbidden}")

        points = 1
        made_first_guess = False
        pre_bust: Optional[int] = None

        while True:
            card = deck.draw()
            if not card:
                self._log(f"  [{player.name}] deck exhausted -> auto-stop with {points}")
                break

            self._log(f"    draw: {card.rank}{card.suit} (rem {deck.remaining()})")

            if card.suit == forbidden:
                pre_bust = points
                self._log(f"    -> BUST! [{player.name}] scores 0 this round.")
                points = 0
                break

            points += 1
            made_first_guess = True

            # Decision to continue/stop
            decision_ctx = {
                "phase": "draw_decision",
                "round_index": round_idx,
                "public": public_state,
                "current_points": points,
                "last_card": {"rank": card.rank, "suit": card.suit},
                "deck_remaining_by_suit": deck.remaining_by_suit(),
            }
            action = player.choose_continue_or_stop(points, decision_ctx)

            if not made_first_guess and action == "stop":
                self._log(f"    [RULE] {player.name} cannot stop before first guess -> forcing continue")
                action = "continue"

            if action not in ("continue", "stop"):
                action = "continue"

            self._log(f"    [{player.name}] points={points}; action={action}")
            if action == "stop":
                break

        self._log(f"  [{player.name}] ends Round {round_idx} with: {points}\n")
        return points, pre_bust

    def _operator_pick_phase(self, round_idx: int, public_state: Dict[str, Any]):
        self._log(f"Operator picks between R{round_idx} & R{round_idx+1}:")
        previous_picks: List[Dict[str, str]] = []
        all_scores = {q.name: list(q.round_scores) for q in self.players}
        
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
            op = "+" if op not in ("+", "x") else op
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

                if self.stats is not None and RoundOutcome is not None:
                    op_ctx = p.ops_between[r - 2] if r > 1 and len(p.ops_between) >= (r - 1) else "+"
                    busted = (score == 0)
                    outcome = RoundOutcome(
                        player=p.name,
                        round_idx=r,
                        points=score,
                        busted=busted,
                        pre_bust_streak=pre_bust,
                        op_context=op_ctx
                    )
                    self.stats.record_round(outcome)

            if r < ROUNDS:
                self._operator_pick_phase(r, public_state)

        results: Dict[str, int] = {p.name: evaluate_expression(p.round_scores, p.ops_between) for p in self.players}

        if self.verbose >= 1:
            print("\n=== FINAL ===")
            for p in self.players:
                line = expr_string_annotated(p.round_scores, p.ops_between, p.pre_bust)
                print(f"{p.name}: {line} = {results[p.name]}")

        best = max(results.values()) if results else None
        leaders = [n for n, v in results.items() if v == best] if best is not None else []
        winner = leaders[0] if len(leaders) == 1 else None

        return winner, results