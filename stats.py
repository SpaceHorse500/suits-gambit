# run.py
from random_player import RandomPlayer
from smart_player import SmartPlayer
from hand_player import HandPlayer
from tactician_player import TacticianPlayer

from game import SuitsGambitGame
from utils import evaluate_expression

import statistics as stats
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Callable, Any
import math

# =========================
# Roster (fresh instances!)
# =========================
def make_players() -> List:
    return [
        RandomPlayer("Random1"),
        RandomPlayer("Random2"),
        SmartPlayer("Smart1"),
        SmartPlayer("Smart2"),
        HandPlayer("Hand1"),
        HandPlayer("Hand2"),
        TacticianPlayer("Tac1"),
        TacticianPlayer("Tac2"),
    ]

# =========================
# Basic stats helpers
# =========================
def iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    q1 = xs[len(xs)//4]
    q3 = xs[(3*len(xs))//4]
    return float(q3 - q1)

def wilson_ci(phat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    center = (phat + (z*z)/(2*n)) / (1 + z*z/n)
    margin = (z / (1 + z*z/n)) * math.sqrt((phat*(1-phat)/n) + (z*z)/(4*n*n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)

def binom_sign_test_pvalue(wins: int, n: int) -> float:
    """Two-sided sign test p using normal approx (fine at n>=30)."""
    if n == 0:
        return 1.0
    p = 0.5
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    z = (wins - mu) / sigma if sigma > 0 else 0.0
    def phi(t): return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    return 2 * (1 - phi(abs(z)))

def cliffs_delta(a: List[float], b: List[float]) -> float:
    # Pairwise (zip) variant, matches your original approach
    gt = sum(1 for x, y in zip(a, b) if x > y)
    lt = sum(1 for x, y in zip(a, b) if x < y)
    n = gt + lt
    return 0.0 if n == 0 else (gt - lt) / n

def zeroed_multiplicative_block(scores: List[int], ops: List[str]) -> bool:
    """True if any ×-block contains a zero (× precedence)."""
    cur_zero = (scores[0] == 0)
    for i, op in enumerate(ops, start=1):
        if op == "x":
            cur_zero = cur_zero or (scores[i] == 0)
        else:  # '+'
            if cur_zero:
                return True
            cur_zero = (scores[i] == 0)
    return cur_zero

# =========================
# Sim with telemetry (extended)
# =========================
def simulate(num_games: int, player_factory: Callable[[], List], game_verbose: int = 0):
    totals_hist: Dict[str, List[int]] = defaultdict(list)
    wins: Dict[str, int] = Counter()
    ties = 0

    bust_rounds = Counter()
    pre_bust_sum = Counter()
    rounds_played = Counter()

    stop_at_two = Counter()
    plus_count = Counter()
    times_count = Counter()
    x_after_x = Counter()
    zeroed_block_games = Counter()

    # NEW: generic stop@K stats
    stop_counts = Counter()             # (name, K) -> count  (banked stops only)
    stop_counts_by_op = Counter()       # (name, op, K) -> count ; op in {'+','x'}
    stop_counts_by_round = Counter()    # (name, round_idx, K) -> count

    per_game_totals: List[Dict[str, int]] = []

    for _ in range(num_games):
        players = player_factory()
        game = SuitsGambitGame(players, verbose=game_verbose)  # keep games quiet for speed
        winner, results = game.play()

        per_game_totals.append(dict(results))

        if winner is None:
            ties += 1
        else:
            wins[winner] += 1

        for p in players:
            name = p.name
            totals_hist[name].append(results[name])

            # Per-round stats
            for r, s in enumerate(p.round_scores):
                rounds_played[name] += 1
                # Bust tracking + pre-bust
                if s == 0:
                    bust_rounds[name] += 1
                    pb = getattr(p, "pre_bust", [None]*5)[r]
                    if pb is not None:
                        pre_bust_sum[name] += pb
                # stop@2 legacy metric
                if s == 2:
                    stop_at_two[name] += 1
                # NEW: stop@K generic (only for banks)
                if s > 0:
                    K = s
                    stop_counts[(name, K)] += 1
                    # operator context for this round is op chosen between (r)->(r+1)
                    # i.e., op BEFORE this round r+1 is p.ops_between[r-1]
                    op_ctx = "+"
                    if r > 0 and r - 1 < len(p.ops_between):
                        op_ctx = p.ops_between[r - 1] if p.ops_between[r - 1] in ("+", "x") else "+"
                    stop_counts_by_op[(name, op_ctx, K)] += 1
                    stop_counts_by_round[(name, r + 1, K)] += 1  # rounds are 1-indexed in report

            # Ops usage and patterns
            ops = list(p.ops_between)
            plus_count[name] += sum(1 for o in ops if o == '+')
            times_count[name] += sum(1 for o in ops if o == 'x')
            for a, b in zip(ops, ops[1:]):
                if a == 'x' and b == 'x':
                    x_after_x[name] += 1

            if zeroed_multiplicative_block(p.round_scores, ops):
                zeroed_block_games[name] += 1

    return {
        "totals_hist": totals_hist,
        "wins": wins,
        "ties": ties,
        "per_game_totals": per_game_totals,
        "bust_rounds": bust_rounds,
        "pre_bust_sum": pre_bust_sum,
        "rounds_played": rounds_played,
        "stop_at_two": stop_at_two,
        "plus_count": plus_count,
        "times_count": times_count,
        "x_after_x": x_after_x,
        "zeroed_block_games": zeroed_block_games,
        "num_games": num_games,
        # NEW: extended stop stats
        "stop_counts": stop_counts,                     # (name, K)
        "stop_counts_by_op": stop_counts_by_op,         # (name, op, K)
        "stop_counts_by_round": stop_counts_by_round,   # (name, round, K)
    }

# =========================
# Behavior similarity
# =========================
def standardize(features: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # z-score each feature across bots
    keys = list(next(iter(features.values())).keys()) if features else []
    out = {n: {} for n in features}
    for k in keys:
        vals = [features[n][k] for n in features]
        m = sum(vals)/len(vals)
        sd = (sum((v-m)**2 for v in vals)/len(vals))**0.5
        for n in features:
            out[n][k] = 0.0 if sd == 0 else (features[n][k]-m)/sd
    return out

def behavior_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    ks = a.keys()
    return sum((a[k]-b[k])**2 for k in ks) ** 0.5

def behavior_features(sim: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    names = sorted(sim["totals_hist"].keys())
    feats: Dict[str, Dict[str, float]] = {}
    ng = sim["num_games"]
    pairs_per_game = 3  # 4 ops → 3 adjacent pairs
    for n in names:
        rounds = max(1, sim["rounds_played"][n])
        ops_total = sim["plus_count"][n] + sim["times_count"][n]
        x_pct = sim["times_count"][n] / ops_total if ops_total > 0 else 0.0
        xchain_rate = sim["x_after_x"][n] / (ng * pairs_per_game) if ng > 0 else 0.0
        bust_rate = sim["bust_rounds"][n] / rounds
        stop2_rate = sim["stop_at_two"][n] / rounds
        zeroed_rate = sim["zeroed_block_games"][n] / ng if ng > 0 else 0.0
        scores = sim["totals_hist"][n]
        avg_pb = (sim["pre_bust_sum"][n] / sim["bust_rounds"][n]) if sim["bust_rounds"][n] else 0.0
        feats[n] = {
            "bust_rate": bust_rate,
            "avg_pre_bust": avg_pb,
            "stop2_rate": stop2_rate,
            "x_pct": x_pct,
            "xchain_rate": xchain_rate,
            "zeroed_rate": zeroed_rate,
            "mean_total": stats.mean(scores),
            "sd_total": stats.pstdev(scores),
            "median_total": stats.median(scores),
        }
    return feats

def print_behavior_similarity(sim: Dict[str, Any]):
    feats = behavior_features(sim)
    zfeats = standardize(feats)
    names = sorted(zfeats.keys())

    # Pairwise distances + verdict
    print("\n— Behavior similarity —")
    SIM_THRESH = 1.20  # lower = more similar
    similar_pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            d = behavior_distance(zfeats[a], zfeats[b])
            verdict = "SIMILAR" if d < SIM_THRESH else "different"
            print(f"{a} vs {b}: {verdict} (dist={d:.2f})")
            if verdict == "SIMILAR":
                similar_pairs.append((a,b))

    if not similar_pairs:
        print("No clearly similar pairs by behavior.")
    else:
        print("Similar pairs:", ", ".join([f"{a}~{b}" for a,b in similar_pairs]))

# =========================
# Pairwise performance
# =========================
def pairwise_performance(sim: Dict[str, Any]):
    names = sorted(sim["totals_hist"].keys())
    cols = {n: [g[n] for g in sim["per_game_totals"]] for n in names}

    print("\n— Pairwise performance (row vs col) —")
    header = " " * 12 + " ".join([f"{n:>10s}" for n in names])
    print(header)
    for a in names:
        row = [f"{a:>12s}"]
        for b in names:
            if a == b:
                row.append(f"{'-':>10s}")
                continue
            awins = sum(1 for x, y in zip(cols[a], cols[b]) if x > y)
            aloss = sum(1 for x, y in zip(cols[a], cols[b]) if x < y)
            n = awins + aloss
            wr = awins / n if n else 0.0
            row.append(f"{100*wr:5.1f}%")
        print(" ".join(row))

    print("\n— Pairwise sign-test p (row vs col) & Cliff’s δ —")
    print(header)
    for a in names:
        row = [f"{a:>12s}"]
        for b in names:
            if a == b:
                row.append(f"{'-':>10s}")
                continue
            aw = sum(1 for x, y in zip(cols[a], cols[b]) if x > y)
            al = sum(1 for x, y in zip(cols[a], cols[b]) if x < y)
            n = aw + al
            p = binom_sign_test_pvalue(aw, n if n else 1)
            d = cliffs_delta(cols[a], cols[b])
            row.append(f"{p:6.3g}/{d:3.2f}")
        print(" ".join(row))

    # High-level verdicts (performance)
    print("\n— Performance verdicts —")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            aw = sum(1 for x, y in zip(cols[a], cols[b]) if x > y)
            al = sum(1 for x, y in zip(cols[a], cols[b]) if x < y)
            n = aw + al
            if n == 0:
                print(f"{a} vs {b}: no decision")
                continue
            wr = aw / n
            p = binom_sign_test_pvalue(aw, n)
            d = abs(cliffs_delta(cols[a], cols[b]))
            verdict = "DIFFERENT" if (p < 0.01 and d >= 0.10) else "similar"
            print(f"{a} vs {b}: {verdict} (wr={100*wr:.1f}%, p={p:.3g}, |δ|={d:.2f})")

# =========================
# Helpers to print extended stop@K
# =========================
def _collect_k_set(sim: Dict[str, Any], names: List[str]) -> List[int]:
    """Collect the set of all observed stop K across all players, sorted."""
    kset = set()
    for name in names:
        for (n, K), cnt in sim["stop_counts"].items():
            if n == name and cnt > 0:
                kset.add(K)
    return sorted(kset)

def _fmt_stop_row(sim: Dict[str, Any], name: str, klist: List[int]) -> str:
    parts = []
    for K in klist:
        cnt = sim["stop_counts"][(name, K)]
        parts.append(f"@{K}:{cnt:4d}")
    return " ".join(parts) if parts else "(none)"

def _fmt_stop_by_op_row(sim: Dict[str, Any], name: str, klist: List[int], op: str) -> str:
    parts = []
    for K in klist:
        cnt = sim["stop_counts_by_op"][(name, op, K)]
        parts.append(f"@{K}:{cnt:4d}")
    return " ".join(parts) if parts else "(none)"

def _fmt_stop_by_round_rows(sim: Dict[str, Any], name: str, klist: List[int], rounds: List[int]) -> List[str]:
    rows = []
    for r in rounds:
        parts = []
        for K in klist:
            cnt = sim["stop_counts_by_round"][(name, r, K)]
            if cnt:
                parts.append(f"@{K}:{cnt:3d}")
        row = f"R{r}: " + (" ".join(parts) if parts else "(none)")
        rows.append(row)
    return rows

# =========================
# Main report
# =========================
def print_stats_report(sim: Dict[str, Any]):
    names = sorted(sim["totals_hist"].keys())
    n_games = sim["num_games"]

    print(f"\n=== TOURNAMENT RESULTS ({n_games} games) ===")

    # Explain metrics up front
    print("\nLegend:")
    print("  mean/median/SD/IQR — distribution of FINAL totals per game")
    print("  win% — share of games with the #1 total (ties excluded)")
    print("  bust rate — % of rounds that ended at 0; avg pre-bust — typical streak before bust")
    print("  ops — count of '+' vs '×'; x→x — consecutive '×' picks; zeroed ×-block — % games where any × segment hit 0")
    print("  stop@2 — how often a round finished exactly at 2 points")

    # Per-bot summary
    print("\n— Per-bot summary —")
    for name in names:
        data = sim["totals_hist"][name]
        mean = stats.mean(data)
        med = stats.median(data)
        sd = stats.pstdev(data)
        mn, mx = min(data), max(data)
        iq = iqr(data)
        win_rate = sim["wins"].get(name, 0) / n_games
        print(f"{name:8s} | mean {mean:6.2f} | median {med:5.1f} | sd {sd:6.2f} | IQR {iq:6.1f} | "
              f"min {mn:3d} | max {mx:3d} | win% {100*win_rate:5.1f}")

    print(f"\nTies: {sim['ties']} ({100*sim['ties']/n_games:.1f}% of games)")

    # Behavioral stats with quick interpretations
    print("\n— Behavioral profiles —")
    ng = sim["num_games"]
    pairs_per_game = 3
    for name in names:
        r = max(1, sim["rounds_played"][name])
        busts = sim["bust_rounds"][name]
        pb_sum = sim["pre_bust_sum"][name]
        avg_pb = (pb_sum / busts) if busts else 0.0
        stop2 = sim["stop_at_two"][name]
        plus = sim["plus_count"][name]
        times = sim["times_count"][name]
        ops_total = plus + times
        x_pct = times/ops_total if ops_total else 0.0
        xchain = sim["x_after_x"][name] / (ng * pairs_per_game) if ng else 0.0
        zgames = sim["zeroed_block_games"][name] / ng if ng else 0.0

        print(f"{name:8s} | bust {100*busts/r:5.1f}%  | avg pre-bust {avg_pb:4.2f} | stop@2 {100*stop2/r:5.1f}% "
              f"| ops: +{plus:4d} ×{times:4d} (× {100*x_pct:4.1f}%) | x→x {100*xchain:5.2f}% | zeroed ×-block {100*zgames:5.1f}%")

        # One-liner tag
        style = []
        style.append("additive-leaning" if x_pct < 0.40 else ("multiplier-leaning" if x_pct > 0.60 else "balanced ops"))
        style.append("banks early" if stop2/r > 0.08 else "rarely banks at 2")
        style.append("draw-risky" if busts/r > 0.45 else ("draw-safe" if busts/r < 0.35 else "draw-moderate"))
        print(f"  → style: {', '.join(style)}")

    # Performance tables & verdicts
    pairwise_performance(sim)

    # Behavior similarity (based on profiles, not results)
    print_behavior_similarity(sim)

    # ========== NEW: Extended stop@K reporting ==========
    print("\n— Extended stop counts (banked rounds only) —")
    kset = _collect_k_set(sim, names)
    if not kset:
        print("(no banked rounds recorded)")
    else:
        for name in names:
            row = _fmt_stop_row(sim, name, kset)
            print(f"{name:8s} | {row}")

    print("\n— stop@K by operator context —")
    for name in names:
        if not kset:
            print(f"{name:8s} | (none)")
            continue
        plus_row = _fmt_stop_by_op_row(sim, name, kset, "+")
        times_row = _fmt_stop_by_op_row(sim, name, kset, "x")
        # Show totals in that op context for reference
        ops_total = sim["plus_count"][name] + sim["times_count"][name]
        plus_rounds = sim["plus_count"][name]  # count of '+' picks (i.e., contexts R2..R5)
        times_rounds = sim["times_count"][name]
        print(f"{name:8s} | '+' {plus_row}")
        print(f"{'':8s} | 'x' {times_row}")

    print("\n— stop@K by round —")
    rounds = [1, 2, 3, 4, 5]
    for name in names:
        rows = _fmt_stop_by_round_rows(sim, name, kset, rounds)
        print(f"{name:8s}")
        for line in rows:
            print(f"  {line}")

# =========================
# Run it
# =========================
if __name__ == "__main__":
    NUM_GAMES = 6000   # adjust as you like
    sim = simulate(NUM_GAMES, make_players, game_verbose=0)
    print_stats_report(sim)
