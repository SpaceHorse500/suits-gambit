from __future__ import annotations
from typing import List, Any, Optional, Dict, Union
import os, json, copy
from meta_player import MetaPlayer
from evo_player import EvoPlayer

class ControlPool:
    def __init__(
        self,
        evo_paths: Union[str, List[str]] = ["good_bots/con1.json", "good_bots/con2.json"],  # Now accepts a list
        include_meta: bool = True,
        only: Optional[List[str]] = None,
    ):
        """
        Args:
            evo_paths: Single path (str) or list of paths to JSON genome files.
            include_meta: Include MetaOne in controls.
            only: Optional allowlist of control names (e.g., ["MetaOne", "EvoCtrl1", "EvoCtrl2"]).
        """
        self.evo_paths = [evo_paths] if isinstance(evo_paths, str) else evo_paths
        self.include_meta = include_meta
        self.only = set(only) if only else None
        self._evo_genomes: Dict[str, Dict] = {}  # Cache genomes by filename

    def _load_evo_genome(self, path: str) -> Optional[Dict]:
        """Load a genome from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("genome") if isinstance(data, dict) else data
        except Exception as e:
            print(f"[ga_controls] WARN: Failed to load {path}: {e}")
            return None

    def make(self) -> List[Any]:
        """Return the list of control players."""
        bots: List[Any] = []

        # Add MetaOne if allowed
        if self.include_meta and (self.only is None or "MetaOne" in self.only):
            bots.append(MetaPlayer("MetaOne"))

        # Add EvoPlayers for each genome file
        for i, path in enumerate(self.evo_paths, start=1):
            bot_name = f"EvoCtrl{i}"
            if self.only is not None and bot_name not in self.only:
                continue  # Skip if not in allowlist

            if path not in self._evo_genomes:  # Cache genomes to avoid reloading
                self._evo_genomes[path] = self._load_evo_genome(path)

            genome = self._evo_genomes.get(path)
            if genome:
                bots.append(EvoPlayer(bot_name, genome=copy.deepcopy(genome)))

        return bots