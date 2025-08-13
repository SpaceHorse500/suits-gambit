# ga_genome.py
import copy, json, hashlib
from typing import Any, Dict, List, Optional
from evo_io import DEFAULT_GENOME
from .ga_bounds import iter_paths, get_holder, clamp_to_bounds

class Genome:
    def __init__(self, data: Dict[str, Any] | None = None):
        self.data: Dict[str, Any] = copy.deepcopy(data if data is not None else DEFAULT_GENOME)
        self.parent_ids: Optional[List[str]] = None  # set by GA runner if desired

    # Stable id computed from payload (updates automatically if payload changes)
    @property
    def id(self) -> str:
        s = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2s(s.encode("utf-8"), digest_size=8).hexdigest()

    @classmethod
    def from_default(cls) -> "Genome":
        return cls(copy.deepcopy(DEFAULT_GENOME))

    def clone(self) -> "Genome":
        return Genome(copy.deepcopy(self.data))

    def paths(self) -> List[str]:
        return iter_paths(self.data)

    def get_holder(self, path: str):
        return get_holder(self.data, path)

    def clamp_all(self) -> None:
        for p in self.paths():
            holder, key = self.get_holder(p)
            holder[key] = clamp_to_bounds(p, holder[key])

    def to_json(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Genome":
        return Genome(copy.deepcopy(d))
