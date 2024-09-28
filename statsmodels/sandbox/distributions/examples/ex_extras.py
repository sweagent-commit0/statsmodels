"""

Created on Wed Feb 19 12:39:49 2014

Author: Josef Perktold
"""
import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.extras import SkewNorm_gen, skewnorm, ACSkewT_gen, NormExpan_gen, pdf_moments, ExpTransf_gen, LogTransf_gen
from statsmodels.stats.moment_helpers import mc2mvsk, mnc2mc, mvsk2mnc
if __name__ == '__main__':
    example_n()
    example_T()
    examples_normexpand()
    examples_transf()