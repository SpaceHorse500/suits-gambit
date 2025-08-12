# game.py
from typing import List, Dict, Any, Optional, Tuple
from cards import Deck
from players import BasePlayer
from utils import evaluate_expression, expr_string_annotated

ROUNDS = 5

# Optional import of extra stats
try:
    from stats_extra import StatsCollector, RoundOutcome  # type: ignore
except Exception:
    StatsCollector = None  # type: ignore
    RoundOutcome = None    # type: ignore


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
        # Back-compat with old bool 'verbose'
        if isinstance(verbose, bool):
            self.verbose = 2 if verbose else 0
        else:
            self.verbose = int(verbose)
        if seed is not None:
            import random
            random.seed(seed)
        self.stats = stats

    # Print only for verbose=2
    def _log(self, msg: str):
        if self.verbose >= 2:
            print(msg)

    def _public_state(self) -> Dict[str, Any]:
        return {
            "players_public": [
                {"name": p.name, "scores": list(p.round_scores), "ops": list(p.ops_between)}
                for p in self.players
            ],
            "rounds": ROUNDS,
            "turn_order": [p.name for p in self.players],
        }

    def _play_single_turn(self, player: BasePlayer, round_idx: int) -> Tuple[int, Optional[int]]:
        """
        One player's turn with a fresh deck.

        FREE POINT: player starts at 1 before any guesses.
        Mandatory one guess: cannot stop until after first safe guess.
        Returns (score, pre_bust_streak) where pre_bust_streak is None if no bust occurred.
        """
        deck = Deck()
        deck.reset()

        # Info card
        first = deck.draw()
        assert first is not None
        self._log(f"  [{player.name}] info card: {first.rank}{first.suit} (rem {deck.remaining()})")

        # Declare forbidden suit
        ctx: Dict[str, Any] = {
            "phase": "declare_forbidden",
            "round_index": round_idx,
            "public": self._public_state(),
            "info_card": {"rank": first.rank, "suit": first.suit},
            "deck_remaining_by_suit": deck.remaining_by_suit(),
        }
        forbidden = player.choose_forbidden_suit(first, ctx)
        self._log(f"  [{player.name}] forbidden suit: {forbidden}")

        # Start with the free point
        points = 1
        made_first_guess = False
        pre_bust: Optional[int] = None

        while True:
            card = deck.draw()
            if not card:
                self._log(f"  [{player.name}] deck exhausted → auto-stop with {points}")
                break

            self._log(f"    draw: {card.rank}{card.suit} (rem {deck.remaining()})")

            if card.suit == forbidden:
                pre_bust = points        # record streak before bust
                self._log(f"    → BUST! [{player.name}] scores 0 this round.")
                points = 0
                break

            # Safe guess
            points += 1
            if not made_first_guess:
                made_first_guess = True

            # Decision to continue/stop
            ctx = {
                "phase": "draw_decision",
                "round_index": round_idx,
                "public": self._public_state(),
                "current_points": points,
                "last_card": {"rank": card.rank, "suit": card.suit},
                "deck_remaining_by_suit": deck.remaining_by_suit(),
            }
            action = player.choose_continue_or_stop(points, ctx)

            # Enforce mandatory first guess
            if not made_first_guess and action == "stop":
                self._log(f"    [RULE] {player.name} cannot stop before first guess → forcing continue")
                action = "continue"

            if action not in ("continue", "stop"):
                action = "continue"

            self._log(f"    [{player.name}] points={points}; action={action}")
            if action == "stop":
                break

        self._log(f"  [{player.name}] ends Round {round_idx} with: {points}\n")
        return points, pre_bust

    def _operator_pick_phase(self, round_idx: int):
        if self.verbose >= 2:
            print(f"Operator picks between R{round_idx} & R{round_idx+1}:")
        previous_picks: List[Dict[str, str]] = []
        for p in self.players:
            ctx = {
                "phase": "operator_pick",
                "round_index": round_idx,
                "public": self._public_state(),
                "previous_picks": list(previous_picks),
                "picker": p.name,
            }
            all_scores = {q.name: list(q.round_scores) for q in self.players}
            op = p.choose_operator_between_rounds(
                my_scores=list(p.round_scores),
                all_scores=all_scores,
                previous_picks=list(previous_picks),
                ctx=ctx,
            )
            op = "+" if op not in ("+", "x") else op
            p.ops_between.append(op)
            previous_picks.append({"player": p.name, "op": op})
            if self.verbose >= 2:
                print(f"  → {p.name}: {op}")

    def play(self):
        # Reset & attach pre_bust storage
        for p in self.players:
            p.reset()
            setattr(p, "pre_bust", [])  # type: ignore[attr-defined]

        # Rounds
        for r in range(1, ROUNDS + 1):
            if self.verbose >= 2:
                print(f"\n=== Round {r} ===")
            for p in self.players:
                score, pre_bust = self._play_single_turn(p, r)
                p.round_scores.append(score)
                p.pre_bust.append(pre_bust)  # type: ignore[attr-defined]

                # ----- Extended stats recording -----
                if self.stats is not None and RoundOutcome is not None:
                    # Operator context for THIS round r is the operator chosen BETWEEN (r-1)->r
                    if r == 1 or len(p.ops_between) < (r - 1):
                        op_ctx = "+"
                    else:
                        op_ctx = p.ops_between[r - 2]

                    busted = (score == 0)
                    outcome = RoundOutcome(
                        player=p.name,
                        round_idx=r,
                        points=score,
                        busted=busted,
                        pre_bust_streak=pre_bust,
                        op_context=op_ctx if op_ctx in {"+", "x"} else "+"
                    )
                    self.stats.record_round(outcome)
                # ------------------------------------

            if r < ROUNDS:
                self._operator_pick_phase(r)

        # Final output
        results: Dict[str, int] = {}
        for p in self.players:
            results[p.name] = evaluate_expression(p.round_scores, p.ops_between)

        if self.verbose >= 1:
            print("\n=== FINAL ===")
            for p in self.players:
                line = expr_string_annotated(p.round_scores, p.ops_between, p.pre_bust)  # type: ignore[attr-defined]
                print(f"{p.name}: {line} = {results[p.name]}")

        # Winner / ties
        best = max(results.values()) if results else None
        leaders = [n for n, v in results.items() if v == best] if best is not None else []
        winner = leaders[0] if len(leaders) == 1 else None

        return winner, results
