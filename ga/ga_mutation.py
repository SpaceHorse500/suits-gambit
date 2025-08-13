# ga_mutation.py
import copy, random
from .ga_types import MutateParams
from .ga_bounds import iter_paths, get_holder, clamp_to_bounds, range_for
from .ga_genome import Genome


class Mutator:
    """
    Gaussian-within-bounds mutator.

    Runner-compat change:
      • mutate(self, genome, rng=None, **kwargs) accepts an optional RNG supplied by the runner.
        If not provided, falls back to a seed-based local RNG or global random.
    """
    def __init__(self, params: MutateParams):
        self.params = params

    def mutate(self, genome: Genome, rng: random.Random | None = None, **kwargs) -> Genome:
        rnd = rng or (random.Random(self.params.seed) if self.params.seed is not None else random)
        child = genome.clone()
        for path in iter_paths(child.data):
            holder, key = get_holder(child.data, path)
            val = holder[key]
            if not isinstance(val, (int, float)):
                continue
            if rnd.random() > self.params.per_gene_prob:
                continue

            lo, hi = range_for(path)
            span = hi - lo if hi > lo else 1.0

            # Reset to random within bounds
            if rnd.random() < self.params.reset_prob:
                holder[key] = clamp_to_bounds(path, lo + rnd.random() * span)
                continue

            # Gaussian jitter scaled by bounds span
            sigma = self.params.sigma_frac * span
            holder[key] = clamp_to_bounds(path, val + rnd.gauss(0.0, sigma))

        return child
