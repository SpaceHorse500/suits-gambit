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
        self.reset()

    def reset(self):
        if self._seed is not None:
            random.seed(self._seed)
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        return self.cards.pop() if self.cards else None

    def remaining(self) -> int:
        return len(self.cards)

    def remaining_by_suit(self) -> Dict[str, int]:
        d = {s: 0 for s in SUITS}
        for c in self.cards:
            d[c.suit] += 1
        return d
