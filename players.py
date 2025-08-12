# players.py
from typing import List, Optional, Dict, Any
from cards import SUITS, Card

class BasePlayer:
    def __init__(self, name: str):
        self.name = name
        self.round_scores: List[int] = []
        self.ops_between: List[str] = []

    def reset(self):
        self.round_scores.clear()
        self.ops_between.clear()

    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        raise NotImplementedError

    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        raise NotImplementedError

    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        raise NotImplementedError
