# ga/ga_repository.py
import os
import json
import datetime
from typing import Dict, Any, Optional
from .ga_types import Fitness

class BotRepository:
    """
    Saves the best genome per generation to bots/<gen>_<shortid>_<timestamp>.json
    and appends a row to bots/manifest.csv (includes bot_id).
    """
    def __init__(self, root: str = "bots"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                f.write("timestamp,generation,bot_id,median,win_rate,max,min,bust_rate,mean,sd,seed,filename\n")

    def save_best(
        self,
        genome_dict: Dict[str, Any],
        fitness: Fitness,
        gen_idx: int,
        eval_seed: int,
        bot_id: Optional[str] = None
    ) -> str:
        # Resolve a stable ID
        uid = bot_id or genome_dict.get("uid") or genome_dict.get("id") or "unknown"
        short_uid = str(uid)[:8]

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"gen{gen_idx:03d}_{short_uid}_{ts}.json"
        fpath = os.path.join(self.root, fname)

        payload = {
            "meta": {
                "generation": gen_idx,
                "timestamp": ts,
                "eval_seed": eval_seed,
                "bot_id": uid,
                "fitness": {
                    "median": getattr(fitness, "median", 0.0),
                    "win_rate": getattr(fitness, "win_rate", 0.0),
                    "max_score": getattr(fitness, "max_score", 0),
                    "min_score": getattr(fitness, "min_score", 0),
                    "bust_rate": (getattr(fitness, "diagnostics", {}) or {}).get("bust_rate"),
                    "mean": (getattr(fitness, "diagnostics", {}) or {}).get("mean"),
                    "sd": (getattr(fitness, "diagnostics", {}) or {}).get("sd"),
                    # pass through quartiles if present
                    "q1": (getattr(fitness, "diagnostics", {}) or {}).get("q1"),
                    "q3": (getattr(fitness, "diagnostics", {}) or {}).get("q3"),
                    "median_observed": (getattr(fitness, "diagnostics", {}) or {}).get("median_observed"),
                },
            },
            "genome": genome_dict,
        }

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        diag = getattr(fitness, "diagnostics", {}) or {}
        bust = float(diag.get("bust_rate", 0.0))
        mean = float(diag.get("mean", 0.0))
        sd = float(diag.get("sd", 0.0))

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(",".join([
                ts,
                str(gen_idx),
                str(uid),
                f"{getattr(fitness, 'median', 0.0):.4f}",
                f"{getattr(fitness, 'win_rate', 0.0):.6f}",
                str(getattr(fitness, "max_score", 0)),
                str(getattr(fitness, "min_score", 0)),
                f"{bust:.6f}",
                f"{mean:.6f}",
                f"{sd:.6f}",
                str(eval_seed),
                fname,
            ]) + "\n")

        return fpath
