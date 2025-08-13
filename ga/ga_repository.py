# ga_repository.py
import os, json, datetime
from typing import Dict, Any, Optional
from .ga_types import Fitness

class BotRepository:
    """
    ID-based repo:
      - Canonical file per bot at:  bots/<ID>/latest.json  (overwritten each save)
      - Optional snapshot per generation at: bots/<ID>/gen_<###>_<timestamp>.json
      - A manifest row is appended for each save (includes the bot ID).

    Expected: genome_dict contains an 'id' field (e.g., short hex like "8b7d0509").
    """
    def __init__(self, root: str = "bots", keep_snapshots: bool = True):
        self.root = root
        self.keep_snapshots = keep_snapshots
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                # include id up front
                f.write(
                    "id,timestamp,generation,median,win_rate,max,min,bust_rate,mean,sd,seed,filename\n"
                )

    @staticmethod
    def _extract_id(genome_dict: Dict[str, Any]) -> str:
        gid = (
            genome_dict.get("id")
            or genome_dict.get("genome_id")
            or genome_dict.get("uid")
        )
        if not gid:
            raise ValueError(
                "BotRepository.save_best: genome_dict is missing an 'id' (or 'genome_id'/'uid')."
            )
        return str(gid)

    def save_best(
        self,
        genome_dict: Dict[str, Any],
        fitness: Fitness,
        gen_idx: int,
        eval_seed: int,
    ) -> str:
        gid = self._extract_id(genome_dict)
        bot_dir = os.path.join(self.root, gid)
        os.makedirs(bot_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Canonical "latest" file (overwritten)
        latest_path = os.path.join(bot_dir, "latest.json")

        # Optional snapshot per generation for history
        snapshot_name = f"gen_{gen_idx:03d}_{ts}.json"
        snapshot_path = os.path.join(bot_dir, snapshot_name)

        # Build payload (include id in meta for convenience)
        payload = {
            "meta": {
                "id": gid,
                "generation": gen_idx,
                "timestamp": ts,
                "eval_seed": eval_seed,
                "fitness": {
                    "median": fitness.median,
                    "win_rate": fitness.win_rate,
                    "max_score": fitness.max_score,
                    "min_score": fitness.min_score,
                    # common diagnostics (others will still be in diagnostics below if needed)
                    "bust_rate": fitness.diagnostics.get("bust_rate"),
                    "mean": fitness.diagnostics.get("mean"),
                    "sd": fitness.diagnostics.get("sd"),
                    "q1": fitness.diagnostics.get("q1"),
                    "q3": fitness.diagnostics.get("q3"),
                    "iqr": (
                        (fitness.diagnostics.get("q3") - fitness.diagnostics.get("q1"))
                        if (isinstance(fitness.diagnostics.get("q1"), (int, float))
                            and isinstance(fitness.diagnostics.get("q3"), (int, float)))
                        else None
                    ),
                },
            },
            "genome": genome_dict,
            "diagnostics": fitness.diagnostics,  # keep everything verbatim
        }

        # Write latest
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Optionally keep snapshot
        if self.keep_snapshots:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        # Append manifest row
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(
                ",".join(
                    [
                        gid,
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
                        # store relative path to the canonical file
                        os.path.join(gid, "latest.json").replace("\\", "/"),
                    ]
                )
                + "\n"
            )

        return latest_path
