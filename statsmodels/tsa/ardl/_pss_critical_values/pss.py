from itertools import product
import os
import numpy as np
PATH = os.environ.get('PSS_PATH', '..')
seed = [3957597042, 2709280948, 499296859, 1555610991, 2390531900, 2160388094, 4098495866, 47221919]
ss = np.random.SeedSequence(seed)
k = list(range(1, 11))
case = list(range(1, 6))
i1 = [True, False]
block_id = list(range(32))
params = list(product(k, case, i1, block_id))
seeds = ss.generate_state(8 * len(params)).reshape((-1, 8)).tolist()
configs = []
for _s, (_k, _case, _i1, _block_id) in zip(seeds, params):
    configs.append({'seed': _s, 'k': _k, 'case': _case, 'i1': _i1, 'block_id': _block_id, 'path': PATH})
if __name__ == '__main__':
    from joblib import Parallel, delayed
    Parallel(n_jobs=10)((delayed(pss_block)(**c) for c in configs))