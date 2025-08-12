# tactician_player.py
import random
from typing import List, Dict, Any, Optional
from cards import SUITS, Card
from players import BasePlayer


class TacticianPlayer(BasePlayer):
    """
    TacticianPlayer — safer forbidden-suit logic, EV-ish draw policy, and calibrated × appetite.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._forbidden: Optional[str] = None

    # -------- Forbidden suit (stats 60% / match 20% / random 20%) --------
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        r = random.random()
        rem: Dict[str, int] = ctx.get("deck_remaining_by_suit") or {}

        if r < 0.60 and rem and all(s in rem for s in SUITS):
            # Stats: pick from suit(s) with FEWEST remaining (MOST revealed)
            min_left = min(rem[s] for s in SUITS)
            fewest = [s for s in SUITS if rem[s] == min_left]
            choice = first_revealed.suit if first_revealed.suit in fewest else random.choice(fewest)
        elif r < 0.80:
            choice = first_revealed.suit
        else:
            choice = random.choice(SUITS)

        self._forbidden = choice
        return choice

    # -------- Continue / Stop (deck-odds + operator aware) --------
    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        last_op = self.ops_between[-1] if self.ops_between else "+"

        rem_by = ctx.get("deck_remaining_by_suit") or {}
        remaining = sum(rem_by.values()) if rem_by else 0
        forb = self._forbidden
        jitter = random.uniform(-0.02, 0.02)

        # Fallback if stats missing
        if remaining <= 0 or not rem_by or forb not in rem_by:
            if last_op == "x":
                if current_points < 3:
                    return "continue"
                base, inc = 0.66, 0.06
            else:
                if current_points < 3:
                    return "continue"
                base, inc = 0.28, 0.14
            n_correct = max(0, current_points - 1)
            stop_prob = min(0.99, base + inc * max(0, n_correct - 1))
            return "stop" if random.random() < stop_prob else "continue"

        forbidden_left = rem_by.get(forb, 0)
        p_bust = forbidden_left / remaining if remaining > 0 else 1.0

        if last_op == "x":
            # Build × segments when safe
            if current_points < 3 and p_bust <= 0.33:
                return "continue"
            if current_points < 4 and p_bust <= 0.15:
                return "continue"
            p_thresh = 0.85 / (current_points + 1)
            return "stop" if p_bust >= (p_thresh + jitter) else "continue"

        # '+' context
        if current_points < 3:
            return "continue"
        if p_bust <= 0.16:
            return "continue"
        p_thresh = 0.95 / (current_points + 1)
        return "stop" if p_bust >= (p_thresh + jitter) else "continue"

    # -------- Operator between rounds (score-aware × appetite) --------
    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        last_round_score = my_scores[-1] if my_scores else 0
        if last_round_score == 0:
            return "+"

        prev_op = self.ops_between[-1] if self.ops_between else None
        if prev_op == "x":
            if last_round_score >= 6:
                repeat_prob = 0.30
            elif last_round_score >= 4:
                repeat_prob = 0.18
            else:
                repeat_prob = 0.10
            return "x" if random.random() < repeat_prob else "+"

        if last_round_score >= 6:
            x_prob = 0.62
        elif last_round_score >= 4:
            x_prob = 0.48
        else:
            x_prob = 0.32

        return "x" if random.random() < x_prob else "+"
