# ga_controls.py
from __future__ import annotations
from typing import List, Any, Optional
import os, json, copy

# keep these imports if you still use them elsewhere
from random_player import RandomPlayer   # noqa: F401
from smart_player import SmartPlayer     # noqa: F401
from hand_player import HandPlayer       # noqa: F401

from meta_player import MetaPlayer
from evo_player import EvoPlayer


class ControlPool:
    def __init__(
        self,
        evo_path: str = "good_bots/con1.json",
        include_meta: bool = True,
        only: Optional[List[str]] = None,
    ):
        """
        include_meta: include MetaOne in controls
        only: optional allowlist of control names to return, e.g. ["EvoCtrl1"] or ["MetaOne","EvoCtrl1"]
        evo_path: path to JSON file containing either:
                  - {"genome": {...}}  (saved by repo)
                  - {...}              (raw genome)
        """
        self.evo_path = evo_path
        self.include_meta = include_meta
        self.only = set(only) if only else None
        self._evo_genome = None  # cached after first successful load

    def _load_evo_genome(self):
        if self._evo_genome is not None:
            return self._evo_genome

        path = self.evo_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("genome"), dict):
                genome = data["genome"]
            elif isinstance(data, dict):
                genome = data
            else:
                print(f"[ga_controls] WARN: {path} did not contain a dict genome.")
                genome = None
        except FileNotFoundError:
            print(f"[ga_controls] WARN: evo control file not found: {path}")
            genome = None
        except Exception as e:
            print(f"[ga_controls] WARN: failed to load evo control from {path}: {e}")
            genome = None

        self._evo_genome = genome
        return genome

    def make(self) -> List[Any]:
        """
        Return the list of control players for a table.
        Called frequently by evaluators, so we instantiate new player objects
        but reuse the cached genome payload.
        """
        bots: List[Any] = []

        if self.include_meta and (self.only is None or "MetaOne" in self.only):
            bots.append(MetaPlayer("MetaOne"))

        genome = self._load_evo_genome()
        if genome is not None and (self.only is None or "EvoCtrl1" in self.only):
            # pass a deep copy to avoid any accidental in-game mutation
            bots.append(EvoPlayer("EvoCtrl1", genome=copy.deepcopy(genome)))

        return bots
