# new_runner.py
import time
from random_player import RandomPlayer
from smart_player import SmartPlayer
from game import SuitsGambitGame

def run_simulation(num_players, num_games=5000):
    """
    Simulates a number of games for a given number of players.
    Returns the elapsed time.
    """
    start_time = time.perf_counter()

    for _ in range(num_games):
        # Create a mix of RandomPlayer and SmartPlayer instances
        players = [RandomPlayer(f"Random{i}") for i in range(num_players // 2)]
        players += [SmartPlayer(f"Smart{i}") for i in range(num_players - (num_players // 2))]

        game = SuitsGambitGame(players, verbose=False)
        game.play()

    end_time = time.perf_counter()
    return end_time - start_time

if __name__ == "__main__":
    print("--- Suits Gambit Game Simulation Timing ---")
    results = {}
    
    # Loop from 1 to 50 players
    for x in range(1, 51):
        # We need at least two players to play a game
        if x < 2:
            print(f"Skipping {x} players as the game requires a minimum of 2.")
            continue

        print(f"Simulating 5000 games with {x} players...")
        elapsed_time = run_simulation(x)
        results[x] = elapsed_time
        print(f"Completed in {elapsed_time:.4f} seconds.\n")

    print("\n--- Summary of Results ---")
    for num_players, duration in results.items():
        print(f"Players: {num_players:2d} | Time: {duration:.4f} seconds")