import random
from players import *

class RandomPlayer(BasePlayer):
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        return random.choice(SUITS)

    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        if current_points < 1:
            return "continue"
        return random.choice(["continue", "stop"])

    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        """
        Picks '+' or 'x' randomly, but avoids 'x' if the just-finished round score is 0.
        """
        last_round_score = my_scores[-1] if my_scores else 0
        if last_round_score == 0:
            return "+"  # never multiply from 0
        return random.choice(["+", "x"])