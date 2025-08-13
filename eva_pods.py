#!/usr/bin/env python3
# eva_pods.py
from __future__ import annotations
import argparse, copy, json, math, os, random
from typing import Any, Dict, List, Tuple

# --- Pod evaluator (from earlier message) ---
from ga.pod_evaluator import PodEloEvaluator

# Try to use your Genome class if it exists, but don't require .random()
try:
    from ga.ga_genome import Genome  # only for type hints / optional usage
except Exception:
    Genome = Any  # type: ignore


# --- A tiny wrapper so EvoPlayer can consume .to_json() as usual ---
class SimpleGenome:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    def to_json(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"))


def _load_baseline(path: str = "weights.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put a baseline genome file there "
            "(your repo already has weights.json)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _jitter_number(key: str, val: float, rng: random.Random, frac: float) -> float:
    """Gaussian jitter by (frac * scale). Clamps 'prob-like' keys to [0,1]."""
    scale = abs(val) + 1e-6
    new_val = val + rng.gauss(0.0, frac * scale)

    # Heuristics: clamp some probability-like knobs into [0,1]
    k = key.lower()
    if any(s in k for s in ("prob", "repeat", "prefer")):
        new_val = max(0.0, min(1.0, new_val))

    # Mildly clamp tiny thresholds from going very negative
    if "threshold" in k:
        new_val = max(-10.0, min(10.0, new_val))

    return new_val


def _jitter_int(key: str, val: int, rng: random.Random, step: int = 1, lo: int = 0, hi: int = 10) -> int:
    """Small integer tweaks, clamped to a reasonable band."""
    delta = rng.choice([-1, 0, +1])
    out = int(val + delta * step)
    return max(lo, min(hi, out))


def _jitter_dict(d: Dict[str, Any], rng: random.Random, frac: float) -> Dict[str, Any]:
    """Deep-copy + jitter numbers; keep structure & keys the same."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _jitter_dict(v, rng, frac)
        elif isinstance(v, (list, tuple)):
            # Jitter numeric elements but preserve length/order
            new_list = []
            for idx, el in enumerate(v):
                if isinstance(el, (int, float)):
                    j = _jitter_number(f"{k}[{idx}]", float(el), rng, frac)
                    # restore int where it was int-ish
                    if isinstance(el, int):
                        j = int(round(j))
                    new_list.append(j)
                else:
                    new_list.append(el)
            out[k] = type(v)(new_list)
        elif isinstance(v, float):
            out[k] = _jitter_number(k, v, rng, frac)
        elif isinstance(v, int):
            out[k] = _jitter_int(k, v, rng, lo=0, hi=12)
        else:
            out[k] = v
    return out


def build_population_from_weights(
    size: int,
    baseline_path: str = "weights.json",
    seed: int = 1234,
    jitter_frac: float = 0.25,
) -> List[SimpleGenome]:
    """
    Make 'size' genomes by jittering the baseline weights.json.
    jitter_frac ~ 0.25 = 25% Gaussian noise relative to |value|.
    """
    rng = random.Random(seed)
    base = _load_baseline(baseline_path)
    pop: List[SimpleGenome] = []
    for _ in range(size):
        g = _jitter_dict(base, rng, jitter_frac)
        pop.append(SimpleGenome(g))
    return pop


def maybe_use_genome_random(size: int) -> List[Any]:
    """
    If your Genome class has a random-like constructor, you can use it here.
    Otherwise we fall back to jittering weights.json.
    """
    # Try common patterns without hard-failing.
    ctor_names = ["random", "rand", "sample", "from_random"]
    for name in ctor_names:
        if hasattr(Genome, name) and callable(getattr(Genome, name)):
            ctor = getattr(Genome, name)
            return [ctor() for _ in range(size)]  # type: ignore
    # Fallback to weights.json jitter
    return build_population_from_weights(size)


def main():
    ap = argparse.ArgumentParser(description="Evaluate genomes in 5-player pods (Elo-based).")
    ap.add_argument("--pop", type=int, default=24, help="Population size (genomes)")
    ap.add_argument("--rounds", type=int, default=120, help="Scheduling rounds (pods per round)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--pod-size", type=int, default=5, help="Players per game (default 5)")
    ap.add_argument("--k", type=float, default=24.0, help="Elo K-factor")
    ap.add_argument("--verbose-game", type=int, default=0, help="0/1/2 matches your engine verbosity")
    ap.add_argument("--jitter", type=float, default=0.25, help="If using weights.json jitter, relative stddev")
    args = ap.parse_args()

    # Build population robustly
    try:
        pop = maybe_use_genome_random(args.pop)
    except Exception:
        # Guaranteed path: jitter weights.json
        pop = build_population_from_weights(args.pop, jitter_frac=args.jitter, seed=args.seed)

    evaluator = PodEloEvaluator(pod_size=args.pod_size, k_factor=args.k)
    fits, control = evaluator.evaluate(
        pop,
        rounds=args.rounds,
        base_seed=args.seed,
        verbose_game=args.verbose_game,
    )

    # Leaderboard by Elo
    leaderboard = sorted(
        [
            (i, f.diagnostics.get("elo", 0.0), f.win_rate,
             f.diagnostics.get("rank_mean", float("inf")),
             f.diagnostics.get("bust_rate", float("nan")),
             f.diagnostics.get("rank_points", 0.0))
            for i, f in enumerate(fits)
        ],
        key=lambda t: t[1],
        reverse=True,
    )[:12]

    print("\n=== POD EVALUATION (pods of {}) ===".format(args.pod_size))
    print("Top by Elo:")
    for i, elo, wr, rmean, br, rpts in leaderboard:
        print(f"Evo{i:02d} | Elo {elo:6.1f} | win% {wr*100:5.2f} | E[rank] {rmean:4.2f} | "
              f"rankPts {rpts:4.2f} | bust% {br*100:5.1f}")

    meta = control["Meta"]
    print(f"\nMeta | Elo {meta.diagnostics['elo']:.1f} | win% {meta.win_rate*100:.2f} "
          f"| median {meta.median:.1f} | games {meta.diagnostics['n_games']}")


if __name__ == "__main__":
    main()
