"""

Created on Mon Dec 10 08:54:02 2012

Author: Josef Perktold
"""
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa
table0 = np.asarray('1 \t0 \t0 \t0 \t0 \t14 \t1.000\n2 \t0 \t2 \t6 \t4 \t2 \t0.253\n3 \t0 \t0 \t3 \t5 \t6 \t0.308\n4 \t0 \t3 \t9 \t2 \t0 \t0.440\n5 \t2 \t2 \t8 \t1 \t1 \t0.330\n6 \t7 \t7 \t0 \t0 \t0 \t0.462\n7 \t3 \t2 \t6 \t3 \t0 \t0.242\n8 \t2 \t5 \t3 \t2 \t2 \t0.176\n9 \t6 \t5 \t2 \t1 \t0 \t0.286\n10 \t0 \t2 \t2 \t3 \t7 \t0.286'.split(), float).reshape(10, -1)
Total = np.asarray('20 \t28 \t39 \t21 \t32'.split('\t'), int)
Pj = np.asarray('0.143 \t0.200 \t0.279 \t0.150 \t0.229'.split('\t'), float)
kappa_wp = 0.21
table1 = table0[:, 1:-1]
print(fleiss_kappa(table1))
table4 = np.array([[20, 5], [10, 15]])
print('res', cohens_kappa(table4), 0.4)
table5 = np.array([[45, 15], [25, 15]])
print('res', cohens_kappa(table5), 0.1304)
table6 = np.array([[25, 35], [5, 35]])
print('res', cohens_kappa(table6), 0.2593)
print('res', cohens_kappa(table6, weights=np.arange(2)), 0.2593)
t7 = np.array([[16, 18, 28], [10, 27, 13], [28, 20, 24]])
print(cohens_kappa(t7, weights=[0, 1, 2]))
table8 = np.array([[25, 35], [5, 35]])
print('res', cohens_kappa(table8))
'\n   Statistic          Value       ASE     95% Confidence Limits\n   ------------------------------------------------------------\n   Simple Kappa      0.3333    0.0814       0.1738       0.4929\n   Weighted Kappa    0.2895    0.0756       0.1414       0.4376\n'
t9 = [[0, 0, 0], [5, 16, 3], [8, 12, 28]]
res9 = cohens_kappa(t9)
print('res', res9)
print('res', cohens_kappa(t9, weights=[0, 1, 2]))
table6a = np.array([[30, 30], [0, 40]])
res = cohens_kappa(table6a)
assert res.kappa == res.kappa_max
print(res.kappa / res.kappa_max)
table10 = [[0, 4, 1], [0, 8, 0], [0, 1, 5]]
res10 = cohens_kappa(table10)
print('res10', res10)
'SAS result for table10\n\n                  Simple Kappa Coefficient\n              --------------------------------\n              Kappa                     0.4842\n              ASE                       0.1380\n              95% Lower Conf Limit      0.2137\n              95% Upper Conf Limit      0.7547\n\n                  Test of H0: Kappa = 0\n\n              ASE under H0              0.1484\n              Z                         3.2626\n              One-sided Pr >  Z         0.0006\n              Two-sided Pr > |Z|        0.0011\n\n                   Weighted Kappa Coefficient\n              --------------------------------\n              Weighted Kappa            0.4701\n              ASE                       0.1457\n              95% Lower Conf Limit      0.1845\n              95% Upper Conf Limit      0.7558\n\n               Test of H0: Weighted Kappa = 0\n\n              ASE under H0              0.1426\n              Z                         3.2971\n              One-sided Pr >  Z         0.0005\n              Two-sided Pr > |Z|        0.0010\n'