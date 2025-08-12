# run.py
from random_player import RandomPlayer
from smart_player import SmartPlayer

from game import SuitsGambitGame

if __name__ == "__main__":
    # Example: 3 random bots (works with 2+ players)
    players = [
        RandomPlayer("P1"),
        RandomPlayer("P2"),
        SmartPlayer("P3"),
        SmartPlayer("P4"),
    ]

    game = SuitsGambitGame(players, verbose=False)
    winner, totals = game.play()

    print("\n--- Totals ---")
    for name, total in totals.items():
        print(f"{name}: {total}")
    print(f"\nWinner: {winner if winner is not None else 'Tie'}")
