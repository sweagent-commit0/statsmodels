"""examples for sandwich estimators of covariance

Author: Josef Perktold

"""
from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw
nobs = 100
kvars = 4
x = np.random.randn(nobs, kvars - 1)
exog = sm.add_constant(x)
params_true = np.ones(kvars)
y_true = np.dot(exog, params_true)
sigma = 0.1 + np.exp(exog[:, -1])
endog = y_true + sigma * np.random.randn(nobs)
self = sm.OLS(endog, exog).fit()
print(self.HC3_se)
print(sw.se_cov(sw.cov_hc3(self)))
assert_almost_equal(sw.se_cov(sw.cov_hc0(self)), self.HC0_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_hc1(self)), self.HC1_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_hc2(self)), self.HC2_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_hc3(self)), self.HC3_se, 15)
print(self.HC0_se)
print(sw.se_cov(sw.cov_hac_simple(self, nlags=0, use_correction=False)))
bse_hac0 = sw.se_cov(sw.cov_hac_simple(self, nlags=0, use_correction=False))
assert_almost_equal(bse_hac0, self.HC0_se, 15)
print(bse_hac0)
bse_hac0c = sw.se_cov(sw.cov_hac_simple(self, nlags=0, use_correction=True))
assert_almost_equal(bse_hac0c, self.HC1_se, 15)
bse_w = sw.se_cov(sw.cov_white_simple(self, use_correction=False))
print(bse_w)
assert_almost_equal(bse_w, self.HC0_se, 15)
bse_wc = sw.se_cov(sw.cov_white_simple(self, use_correction=True))
print(bse_wc)
assert_almost_equal(bse_wc, self.HC1_se, 15)
groups = np.repeat(np.arange(5), 20)
idx = np.nonzero(np.diff(groups))[0].tolist()
groupidx = lzip([0] + idx, idx + [len(groups)])
ngroups = len(groupidx)
print(sw.se_cov(sw.cov_cluster(self, groups)))
print(sw.se_cov(sw.cov_cluster(self, np.ones(len(endog), int), use_correction=False)))
print(sw.se_cov(sw.cov_crosssection_0(self, np.arange(len(endog)))))
groups = np.repeat(np.arange(50), 100 // 50)
print(sw.se_cov(sw.cov_cluster(self, groups)))
groups = np.repeat(np.arange(2), 100 // 2)
print(sw.se_cov(sw.cov_cluster(self, groups)))
'http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt'
'\ntest <- read.table(\n      url(paste("http://www.kellogg.northwestern.edu/",\n            "faculty/petersen/htm/papers/se/",\n            "test_data.txt",sep="")),\n    col.names=c("firmid", "year", "x", "y"))\n'