# ga_controls.py
from typing import List
from random_player import RandomPlayer
from smart_player import SmartPlayer
from hand_player import HandPlayer

class ControlPool:
    def make(self) -> List:
        # Your requested controls
        return [
            RandomPlayer("Random2"),
            SmartPlayer("Smart1"),
            HandPlayer("Hand1"),
        ]
