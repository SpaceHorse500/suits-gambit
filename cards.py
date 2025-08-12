# cards.py
from dataclasses import dataclass
import random
from typing import List, Optional, Dict

SUITS = ["♣", "♦", "♥", "♠"]
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

class Deck:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self.cards: List[Card] = []
        # Store suit counts to avoid re-calculating
        self._remaining_suits: Dict[str, int] = {s: 0 for s in SUITS}
        self.reset()

    def reset(self):
        if self._seed is not None:
            random.seed(self._seed)
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)
        # Recalculate suit counts after reset
        self._remaining_suits = {s: 13 for s in SUITS}

    def draw(self) -> Optional[Card]:
        if not self.cards:
            return None
        card = self.cards.pop()
        self._remaining_suits[card.suit] -= 1
        return card

    def remaining(self) -> int:
        return len(self.cards)

    def remaining_by_suit(self) -> Dict[str, int]:
        # Return the pre-calculated counts instead of iterating
        return self._remaining_suits