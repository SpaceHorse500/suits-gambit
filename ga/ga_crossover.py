# ga_crossover.py
import copy, random
from .ga_types import CrossoverParams
from .ga_bounds import iter_paths, get_holder, clamp_to_bounds, INT_KEYS
from .ga_genome import Genome


class CrossoverOperator:
    """
    Per-gene arithmetic (BLX-style) crossover with clamping.

    Runner-compat change:
      • crossover(self, a, b, rng=None, **kwargs) accepts an optional RNG supplied by the runner.
        If not provided, falls back to a seed-based local RNG or global random.
    """
    def __init__(self, params: CrossoverParams):
        self.params = params

    def crossover(self, a: Genome, b: Genome, rng: random.Random | None = None, **kwargs) -> Genome:
        rnd = rng or (random.Random(self.params.seed) if self.params.seed is not None else random)
        child = a.clone()
        for path in iter_paths(child.data):
            if rnd.random() > self.params.per_gene_prob:
                continue

            hold_c, kc = get_holder(child.data, path)
            hold_a, ka = get_holder(a.data, path)
            hold_b, kb = get_holder(b.data, path)
            va, vb = hold_a[ka], hold_b[kb]

            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                alpha = rnd.uniform(self.params.alpha_min, self.params.alpha_max)
                v = alpha * va + (1 - alpha) * vb
                if path in INT_KEYS:
                    v = round(v)
                hold_c[kc] = clamp_to_bounds(path, v)
            else:
                # Non-numeric genes: copy from parent A (could randomize parent choice if desired)
                hold_c[kc] = copy.deepcopy(va)

        return child
