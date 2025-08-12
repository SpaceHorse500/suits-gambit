# ga_crossover.py
import copy, random
from .ga_types import CrossoverParams
from .ga_bounds import iter_paths, get_holder, clamp_to_bounds, INT_KEYS
from .ga_genome import Genome

class CrossoverOperator:
    def __init__(self, params: CrossoverParams):
        self.params = params

    def crossover(self, a: Genome, b: Genome) -> Genome:
        rnd = random.Random(self.params.seed) if self.params.seed is not None else random
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
                if path in INT_KEYS: v = round(v)
                hold_c[kc] = clamp_to_bounds(path, v)
            else:
                hold_c[kc] = copy.deepcopy(va)
        return child
