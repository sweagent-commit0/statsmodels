"""
This script provides a tutorial on how to use estimate and conduct
inference in an accelerated failure time model using empirical likelihood.

We will be using the Stanford Heart Transplant data

"""
import numpy as np
import statsmodels.api as sm
data = sm.datasets.heart.load()
model = sm.emplike.emplikeAFT(np.log10(data.endog), sm.add_constant(data.exog), data.censors)
fitted = model.fit()
print(fitted.params())
test1 = fitted.test_beta([4], [0])
print(test1)
test2 = fitted.test_beta([-0.05], [1])
print(test2)
ci_beta1 = fitted.ci_beta(1, 0.1, -0.1)
print(ci_beta1)