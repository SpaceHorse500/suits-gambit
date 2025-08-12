import os
import json
import argparse
import random
import statistics as stats
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Callable

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Project imports
from evo_player import EvoPlayer
from game import SuitsGambitGame
from ga.ga_controls import ControlPool

# =========================
# Enhanced Statistics Helpers
# =========================
def iqr(values: List[float]) -> float:
    """Calculate interquartile range"""
    if not values:
        return 0.0
    xs = sorted(values)
    q1 = xs[len(xs)//4]
    q3 = xs[(3*len(xs))//4]
    return float(q3 - q1)

def cliffs_delta(a: List[float], b: List[float]) -> float:
    """Effect size measure for ordinal data"""
    gt = sum(1 for x, y in zip(a, b) if x > y)
    lt = sum(1 for x, y in zip(a, b) if x < y)
    n = gt + lt
    return 0.0 if n == 0 else (gt - lt) / n

def zeroed_multiplicative_block(scores: List[int], ops: List[str]) -> bool:
    """Check if any ×-block contains a zero"""
    cur_zero = (scores[0] == 0)
    for i, op in enumerate(ops, start=1):
        if op == "x":
            cur_zero = cur_zero or (scores[i] == 0)
        else:
            if cur_zero:
                return True
            cur_zero = (scores[i] == 0)
    return cur_zero

# =========================
# Simulation with Advanced Telemetry
# =========================
def simulate_tournament(
    evo_bots: List[Tuple[str, Dict[str, Any]]],
    include_controls: bool = True,
    games: int = 5000,
    seed: int = 123,
    game_verbose: int = 0,
) -> Dict[str, Any]:
    """Run tournament with advanced statistics tracking"""
    # Create players
    evos_players = [EvoPlayer(name, genome=g) for name, g in evo_bots]
    control_players = ControlPool().make() if include_controls else []
    all_players = evos_players + control_players
    
    # Initialize trackers
    totals_hist = defaultdict(list)
    wins = Counter()
    ties = 0
    bust_rounds = Counter()
    pre_bust_sum = Counter()
    rounds_played = Counter()
    stop_at_two = Counter()
    plus_count = Counter()
    times_count = Counter()
    x_after_x = Counter()
    zeroed_block_games = Counter()
    stop_counts = Counter()
    stop_counts_by_op = Counter()
    stop_counts_by_round = Counter()
    per_game_totals = []

    # Run simulations
    for gidx in range(games):
        rng = random.Random(seed + gidx)
        players = list(all_players)
        rng.shuffle(players)

        game = SuitsGambitGame(players, verbose=game_verbose, seed=seed + gidx)
        winner, results = game.play()

        # Record results
        per_game_totals.append(results)
        if winner is None:
            ties += 1
        else:
            wins[winner] += 1

        for p in players:
            name = p.name
            totals_hist[name].append(results[name])
            
            # Round-level statistics
            for r, score in enumerate(p.round_scores):
                rounds_played[name] += 1
                if score == 0:
                    bust_rounds[name] += 1
                    pb = getattr(p, "pre_bust", [None]*5)[r]
                    if pb is not None:
                        pre_bust_sum[name] += pb
                if score == 2:
                    stop_at_two[name] += 1
                
                # Track stop counts
                if score > 0:
                    K = score
                    stop_counts[(name, K)] += 1
                    op_ctx = "+" if r == 0 else p.ops_between[r-1] if r-1 < len(p.ops_between) else "+"
                    stop_counts_by_op[(name, op_ctx, K)] += 1
                    stop_counts_by_round[(name, r+1, K)] += 1

            # Operator statistics
            ops = p.ops_between
            plus_count[name] += sum(1 for o in ops if o == '+')
            times_count[name] += sum(1 for o in ops if o == 'x')
            for a, b in zip(ops, ops[1:]):
                if a == 'x' and b == 'x':
                    x_after_x[name] += 1

            if zeroed_multiplicative_block(p.round_scores, ops):
                zeroed_block_games[name] += 1

    return {
        "summaries": {
            name: {
                "games": games,
                "median": float(stats.median(totals_hist[name])),
                "mean": float(stats.mean(totals_hist[name])),
                "sd": float(stats.pstdev(totals_hist[name])) if len(totals_hist[name]) > 1 else 0.0,
                "win_rate": wins[name] / games,
                "max": max(totals_hist[name]),
                "min": min(totals_hist[name]),
                "bust_rate": bust_rounds[name] / rounds_played[name] if rounds_played[name] else 0.0,
                "iqr": iqr(totals_hist[name]),
                "wins": wins[name],
                "avg_pre_bust": pre_bust_sum[name] / bust_rounds[name] if bust_rounds[name] else 0.0,
                "stop2_rate": stop_at_two[name] / rounds_played[name] if rounds_played[name] else 0.0,
                "x_pct": times_count[name] / (plus_count[name] + times_count[name]) if (plus_count[name] + times_count[name]) else 0.0,
                "xchain_rate": x_after_x[name] / (games * 3),  # 3 op pairs per game
                "zeroed_rate": zeroed_block_games[name] / games,
            }
            for name in totals_hist
        },
        "per_game_totals": per_game_totals,
        "ties": ties,
        "games": games,
        "table_size": len(all_players),
        "baseline": 1.0 / len(all_players) if all_players else 0.0,
        "evo_names": [p.name for p in evos_players],
        "control_names": [p.name for p in control_players],
        "stop_counts": stop_counts,
        "stop_counts_by_op": stop_counts_by_op,
        "stop_counts_by_round": stop_counts_by_round,
    }

# =========================
# Enhanced Reporting with Colors
# =========================
def print_section_header(title: str, color: str = Colors.CYAN):
    """Print a formatted section header"""
    print(f"\n{color}{Colors.BOLD}=== {title.upper()} ==={Colors.END}")

def print_subsection_header(title: str, color: str = Colors.YELLOW):
    """Print a formatted subsection header"""
    print(f"\n{color}{Colors.UNDERLINE}— {title} —{Colors.END}")

def colorize_number(value: float, reverse: bool = False, fmt: str = ".2f") -> str:
    """Color code numbers based on their value, preserving formatting"""
    if isinstance(value, str):
        return value
    
    formatted_value = format(value, fmt)
    
    if reverse:
        # For values where lower is better (like bust rate)
        if value > 0.5: return f"{Colors.RED}{formatted_value}{Colors.END}"
        if value > 0.3: return f"{Colors.YELLOW}{formatted_value}{Colors.END}"
        return f"{Colors.GREEN}{formatted_value}{Colors.END}"
    else:
        # For values where higher is better (like win rate)
        if value > 0.5: return f"{Colors.GREEN}{formatted_value}{Colors.END}"
        if value > 0.3: return f"{Colors.YELLOW}{formatted_value}{Colors.END}"
        return f"{Colors.RED}{formatted_value}{Colors.END}"

def print_advanced_report(result: Dict[str, Any]):
    """Print comprehensive analysis with colorful formatting"""
    summaries = result["summaries"]
    games = result["games"]
    baseline = result["baseline"]
    evo_names = set(result["evo_names"])
    control_names = set(result["control_names"])

    # Rank by win rate
    ranked = sorted(summaries.keys(),
                   key=lambda n: (summaries[n]["win_rate"], summaries[n]["median"], summaries[n]["mean"]),
                   reverse=True)

    print_section_header(f"Advanced Evaluation: {games} games | Table Size={result['table_size']} | Baseline Win≈{100*baseline:.2f}%")

    # Main performance metrics
    print_subsection_header("Performance Summary")
    for name in ranked:
        s = summaries[name]
        win_mult = s['win_rate']/baseline if baseline > 0 else 0.0
        print(f"{Colors.BOLD}{name:24s}{Colors.END} | "
              f"med {colorize_number(s['median'], fmt='5.2f')} | "
              f"avg {colorize_number(s['mean'], fmt='5.2f')} | "
              f"sd {s['sd']:5.2f} | "
              f"win% {colorize_number(100*s['win_rate'], fmt='5.2f')} ({colorize_number(win_mult, fmt='4.2f')}×) | "
              f"max {Colors.BOLD}{s['max']:3d}{Colors.END} | min {s['min']:3d} | "
              f"bust% {colorize_number(100*s['bust_rate'], reverse=True, fmt='5.2f')} | "
              f"IQR {s['iqr']:5.2f}")

    # Behavioral profiles
    print_subsection_header("Behavioral Profiles")
    for name in ranked:
        s = summaries[name]
        
        # Style classification
        style_parts = []
        if s['x_pct'] < 0.4:
            style_parts.append(f"{Colors.BLUE}additive{Colors.END}")
        elif s['x_pct'] > 0.6:
            style_parts.append(f"{Colors.RED}multiplier{Colors.END}")
        else:
            style_parts.append(f"{Colors.GREEN}balanced{Colors.END}")
            
        if s['bust_rate'] > 0.45:
            style_parts.append(f"{Colors.RED}risky{Colors.END}")
        elif s['bust_rate'] < 0.35:
            style_parts.append(f"{Colors.GREEN}safe{Colors.END}")
        else:
            style_parts.append(f"{Colors.YELLOW}moderate{Colors.END}")
            
        if s['stop2_rate'] > 0.08:
            style_parts.append(f"{Colors.CYAN}banks@2{Colors.END}")
        else:
            style_parts.append("rare@2")
        
        print(f"{Colors.BOLD}{name:24s}{Colors.END} | "
              f"bust {colorize_number(100*s['bust_rate'], reverse=True, fmt='5.1f')}% | "
              f"pre-bust {colorize_number(s['avg_pre_bust'], fmt='4.2f')} | "
              f"ops: +/{colorize_number(s['x_pct'], fmt='.1%')}× | "
              f"chains {colorize_number(100*s['xchain_rate'], fmt='4.1f')}% | "
              f"zeroed {colorize_number(100*s['zeroed_rate'], reverse=True, fmt='4.1f')}% | "
              f"style: {', '.join(style_parts)}")

    # Pairwise comparisons
    print_subsection_header("Pairwise Performance")
    cols = {n: [g[n] for g in result["per_game_totals"]] for n in summaries}
    header = " " * 24 + " ".join([f"{Colors.BOLD}{n[:10]:>10s}{Colors.END}" for n in ranked])
    print(header)
    for a in ranked:
        row = [f"{Colors.BOLD}{a:24s}{Colors.END}"]
        for b in ranked:
            if a == b:
                row.append(f"{'-':>10s}")
                continue
            aw = sum(1 for x, y in zip(cols[a], cols[b]) if x > y)
            al = sum(1 for x, y in zip(cols[a], cols[b]) if x < y)
            n = aw + al
            wr = aw / n if n else 0.0
            d = cliffs_delta(cols[a], cols[b])
            
            # Color based on win rate
            if wr > 0.6:
                wr_str = f"{Colors.GREEN}{100*wr:5.1f}%{Colors.END}"
            elif wr < 0.4:
                wr_str = f"{Colors.RED}{100*wr:5.1f}%{Colors.END}"
            else:
                wr_str = f"{Colors.YELLOW}{100*wr:5.1f}%{Colors.END}"
                
            # Color based on effect size
            if abs(d) > 0.4:
                d_str = f"{Colors.GREEN if d > 0 else Colors.RED}{d:+.2f}{Colors.END}"
            elif abs(d) > 0.2:
                d_str = f"{Colors.YELLOW}{d:+.2f}{Colors.END}"
            else:
                d_str = f"{d:+.2f}"
                
            row.append(f"{wr_str}/{d_str}")
        print(" ".join(row))

    # Stop behavior analysis
    print_subsection_header("Stop Behavior Analysis")
    kset = sorted({k for (n, k) in result["stop_counts"] if result["stop_counts"][(n, k)] > 0})
    for name in ranked:
        stops = [result["stop_counts"].get((name, k), 0) for k in kset]
        stop_strs = []
        for k, cnt in zip(kset, stops):
            if cnt > 0:
                if cnt > games * 0.1:  # Highlight frequent stops
                    stop_strs.append(f"@{k}:{Colors.GREEN}{cnt:3d}{Colors.END}")
                else:
                    stop_strs.append(f"@{k}:{cnt:3d}")
        print(f"{Colors.BOLD}{name:24s}{Colors.END} | " + " ".join(stop_strs))

    # Final summary
    print_section_header("Evaluation Complete", Colors.GREEN)
    print(f"\n{Colors.BOLD}Key:{Colors.END}")
    print(f"{Colors.GREEN}Green{Colors.END} = Good performance/safe behavior")
    print(f"{Colors.YELLOW}Yellow{Colors.END} = Moderate performance")
    print(f"{Colors.RED}Red{Colors.END} = Poor performance/risky behavior")
    print(f"{Colors.BLUE}Blue{Colors.END} = Additive playstyle")
    print(f"{Colors.RED}Red{Colors.END} = Multiplier playstyle")
    print(f"{Colors.GREEN}Green{Colors.END} = Balanced playstyle")

# =========================
# Main Execution
# =========================
def load_saved_bots(folder: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load bot genomes from JSON files"""
    items = []
    if not os.path.isdir(folder):
        print(f"{Colors.RED}[WARN] Bots folder not found: {folder}{Colors.END}")
        return items

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                genome = data.get("genome", data)  # Support both wrapped and raw genomes
                if not isinstance(genome, dict):
                    continue
                
                # Extract timestamp from filename
                timestamp = fname.split('_')[-1].split('.')[0]
                name = f"EvoGen{fname[3:6]}@{timestamp}"  # Convert gen005... to EvoGen005@...
                items.append((name, genome))
        except Exception as e:
            print(f"{Colors.YELLOW}[WARN] Skipping {fname}: {e}{Colors.END}")

    return items

def main():
    ap = argparse.ArgumentParser(description="Advanced evaluation of good bots")
    ap.add_argument("--folder", default="good_bots", help="Folder containing good bot JSONs")
    ap.add_argument("--games", type=int, default=5000, help="Number of games to simulate")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--no-controls", action="store_true", help="Exclude control bots")
    args = ap.parse_args()

    print(f"\n{Colors.HEADER}{Colors.BOLD}=== Suits Gambit Bot Evaluator ==={Colors.END}\n")
    
    bots = load_saved_bots(args.folder)
    if not bots:
        print(f"{Colors.RED}No saved bots found in {args.folder}.{Colors.END}")
        return

    print(f"{Colors.GREEN}Loaded {len(bots)} bots from {args.folder}{Colors.END}")

    res = simulate_tournament(
        evo_bots=bots,
        include_controls=not args.no_controls,
        games=args.games,
        seed=args.seed,
        game_verbose=0
    )

    print_advanced_report(res)

if __name__ == "__main__":
    main()