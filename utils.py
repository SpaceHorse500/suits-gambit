# utils.py
from typing import List, Optional

def evaluate_expression(scores: List[int], ops: List[str]) -> int:
    """
    Evaluate with normal precedence (× before +).
    Example: 4 + 0 + 0 + 2 x 4  ->  4 + 0 + 0 + (2*4) = 12
    """
    assert len(scores) == 5 and len(ops) == 4
    total = 0
    cur = scores[0]
    for i, op in enumerate(ops, start=1):
        if op == "x":
            cur *= scores[i]
        else:  # '+'
            total += cur
            cur = scores[i]
    total += cur
    return total

def expr_string(scores: List[int], ops: List[str]) -> str:
    """
    Plain expression string without annotations.
    """
    out = f"{scores[0]}"
    for i, op in enumerate(ops, start=1):
        out += f" {op} {scores[i]}"
    return out

def expr_string_annotated(round_scores, operators, pre_busts):
    """
    Build a string like: 0 + 2 + 4#0 + 2 + 3
    where '4#0' means the player had 4 points before busting to 0.
    """
    parts = []
    for i, score in enumerate(round_scores):
        if score == 0 and pre_busts[i] is not None:
            parts.append(f"[{pre_busts[i]}].0")  # bust annotation
        else:
            parts.append(str(score))
        if i < len(operators):
            parts.append(operators[i])
    return " ".join(parts)

