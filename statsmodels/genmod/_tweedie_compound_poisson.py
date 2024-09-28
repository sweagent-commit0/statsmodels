"""
Private experimental module for miscellaneous Tweedie functions.

References
----------

Dunn, Peter K. and Smyth,  Gordon K. 2001. Tweedie family densities: methods of
    evaluation. In Proceedings of the 16th International Workshop on
    Statistical Modelling, Odense, Denmark, 2–6 July.

Jørgensen, B., Demétrio, C.G.B., Kristensen, E., Banta, G.T., Petersen, H.C.,
    Delefosse, M.: Bias-corrected Pearson estimating functions for Taylor’s
    power law applied to benthic macrofauna data. Stat. Probab. Lett. 81,
    749–758 (2011)

Smyth G.K. and Jørgensen B. 2002. Fitting Tweedie's compound Poisson model to
    insurance claims data: dispersion modelling. ASTIN Bulletin 32: 143–157
"""
import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
if __name__ == '__main__':
    from scipy import stats
    n = stats.poisson.rvs(0.1, size=10000000)
    y = stats.gamma.rvs(0.1, scale=30000, size=10000000)
    y = n * y
    mu = stats.gamma.rvs(10, scale=30, size=10000000)
    import time
    t = time.time()
    out = series_density(y=y, mu=mu, p=1.5, phi=20)
    print('That took {} seconds'.format(time.time() - t))