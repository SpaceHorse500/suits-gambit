# stats_extra.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from collections import defaultdict


@dataclass
class RoundOutcome:
    player: str
    round_idx: int
    points: int                      # final round points (0 if bust)
    busted: bool                     # True if round ended on forbidden suit
    pre_bust_streak: Optional[int]   # points just before bust (None if not busted)
    op_context: str                  # operator BEFORE this round: '+' or 'x'


@dataclass
class StatsCollector:
    # stop@K overall
    stop_at_overall: Dict[int, int] = field(default_factory=dict)
    # stop@K by operator context
    stop_at_by_op: Dict[str, Dict[int, int]] = field(default_factory=lambda: {"+": {}, "x": {}})
    # stop@K by round index
    stop_at_by_round: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # busts
    busts_total: int = 0
    busts_by_round: Dict[int, int] = field(default_factory=dict)
    # pre-bust streak histogram (1,2,3,…) counts
    pre_bust_hist: Dict[int, int] = field(default_factory=dict)

    # per-operator counters (how many rounds occurred under each op context)
    rounds_by_op: Dict[str, int] = field(default_factory=lambda: {"+": 0, "x": 0})

    # store raw outcomes if you want to post-process later
    outcomes: List[RoundOutcome] = field(default_factory=list)

    def record_round(self, outcome: RoundOutcome):
        self.outcomes.append(outcome)

        op = outcome.op_context if outcome.op_context in {"+", "x"} else "+"
        self.rounds_by_op[op] += 1

        if outcome.busted:
            self.busts_total += 1
            self.busts_by_round[outcome.round_idx] = self.busts_by_round.get(outcome.round_idx, 0) + 1
            if outcome.pre_bust_streak:
                self.pre_bust_hist[outcome.pre_bust_streak] = self.pre_bust_hist.get(outcome.pre_bust_streak, 0) + 1
            return

        k = outcome.points  # stopped/banked at K
        self.stop_at_overall[k] = self.stop_at_overall.get(k, 0) + 1

        if outcome.round_idx not in self.stop_at_by_round:
            self.stop_at_by_round[outcome.round_idx] = {}
        self.stop_at_by_round[outcome.round_idx][k] = self.stop_at_by_round[outcome.round_idx].get(k, 0) + 1

        if op not in self.stop_at_by_op:
            self.stop_at_by_op[op] = {}
        self.stop_at_by_op[op][k] = self.stop_at_by_op[op].get(k, 0) + 1

    # ---------- Summaries ----------
    def summary_stop_overall(self) -> Dict[int, int]:
        return dict(sorted(self.stop_at_overall.items()))

    def summary_stop_by_op(self) -> Dict[str, Dict[int, int]]:
        return {op: dict(sorted(kmap.items())) for op, kmap in self.stop_at_by_op.items()}

    def summary_stop_by_round(self) -> Dict[int, Dict[int, int]]:
        return {r: dict(sorted(kmap.items())) for r, kmap in sorted(self.stop_at_by_round.items())}

    def summary_busts(self):
        return {
            "total_busts": self.busts_total,
            "busts_by_round": dict(sorted(self.busts_by_round.items())),
            "pre_bust_hist": dict(sorted(self.pre_bust_hist.items())),
        }

    # ---------- Pretty-printers ----------
    def _fmt_kmap(self, kmap: Dict[int, int]) -> str:
        if not kmap:
            return "{}"
        items = ", ".join(f"{k}: {v}" for k, v in sorted(kmap.items()))
        return "{" + items + "}"

    def pretty_print(self) -> str:
        lines: List[str] = []
        lines.append("=== EXTENDED ROUND-STOP & BUST METRICS ===")
        lines.append("\n-- stop@K overall --")
        lines.append(self._fmt_kmap(self.summary_stop_overall()))

        lines.append("\n-- stop@K by operator --")
        for op, km in self.summary_stop_by_op().items():
            lines.append(f"{op}: {self._fmt_kmap(km)}  (rounds: {self.rounds_by_op.get(op,0)})")

        lines.append("\n-- stop@K by round --")
        for rnd, km in self.summary_stop_by_round().items():
            lines.append(f"R{rnd}: {self._fmt_kmap(km)}")

        b = self.summary_busts()
        lines.append("\n-- busts & pre-bust --")
        lines.append(f"total_busts: {b['total_busts']}")
        lines.append(f"busts_by_round: {self._fmt_kmap(b['busts_by_round'])}")
        lines.append(f"pre_bust_hist: {self._fmt_kmap(b['pre_bust_hist'])}")

        return "\n".join(lines)
