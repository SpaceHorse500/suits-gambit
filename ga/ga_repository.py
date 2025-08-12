# ga_repository.py
import os, json, datetime
from typing import Dict, Any
from .ga_types import Fitness

class BotRepository:
    def __init__(self, root: str = "bots"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                f.write("timestamp,generation,median,win_rate,max,min,bust_rate,mean,sd,seed,filename\n")

    def save_best(self, genome_dict: Dict[str, Any], fitness: Fitness, gen_idx: int, eval_seed: int) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"gen{gen_idx:03d}_med{fitness.median:.2f}_win{int(100*fitness.win_rate):02d}_max{fitness.max_score}_min{fitness.min_score}_{ts}.json"
        fpath = os.path.join(self.root, fname)
        payload = {
            "meta": {
                "generation": gen_idx,
                "timestamp": ts,
                "eval_seed": eval_seed,
                "fitness": {
                    "median": fitness.median,
                    "win_rate": fitness.win_rate,
                    "max_score": fitness.max_score,
                    "min_score": fitness.min_score,
                    "bust_rate": fitness.diagnostics.get("bust_rate"),
                    "mean": fitness.diagnostics.get("mean"),
                    "sd": fitness.diagnostics.get("sd"),
                }
            },
            "genome": genome_dict
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(",".join([
                ts,
                str(gen_idx),
                f"{fitness.median:.2f}",
                f"{fitness.win_rate:.4f}",
                str(fitness.max_score),
                str(fitness.min_score),
                f"{fitness.diagnostics.get('bust_rate', 0.0):.4f}",
                f"{fitness.diagnostics.get('mean', 0.0):.4f}",
                f"{fitness.diagnostics.get('sd', 0.0):.4f}",
                str(eval_seed),
                fname
            ]) + "\n")
        return fpath
