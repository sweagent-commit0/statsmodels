"""Example: statsmodels.discretemod
"""
from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary import summary_params_2d, summary_params_2dflat, table_extend
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=100)
exog_names = [anes_data.exog_name[i] for i in [0, 2] + lrange(5, 8)] + ['const']
endog_names = [anes_data.endog_name + '_%d' % i for i in np.unique(mlogit_res.model.endog)[1:]]
print('\n\nMultinomial')
print(summary_params_2d(mlogit_res, extras=['bse', 'tvalues'], endog_names=endog_names, exog_names=exog_names))
tables, table_all = summary_params_2dflat(mlogit_res, endog_names=endog_names, exog_names=exog_names, keep_headers=True)
tables, table_all = summary_params_2dflat(mlogit_res, endog_names=endog_names, exog_names=exog_names, keep_headers=False)
print('\n\n')
print(table_all)
print('\n\n')
print('\n'.join((str(t) for t in tables)))
at = table_extend(tables)
print(at)
print('\n\n')
print(mlogit_res.summary())
print(mlogit_res.summary(yname='PID'))
endog_names = [anes_data.endog_name + '=%d' % i for i in np.unique(mlogit_res.model.endog)[1:]]
print(mlogit_res.summary(yname='PID', yname_list=endog_names, xname=exog_names))
" #trying pickle\nimport pickle\n\n#copy.deepcopy(mlogit_res)  #raises exception: AttributeError: 'ResettableCache' object has no attribute '_resetdict'\nmnl_res = mlogit_mod.fit(method='bfgs', maxiter=100)\nmnl_res.cov_params()\n#mnl_res.model.endog = None\n#mnl_res.model.exog = None\npickle.dump(mnl_res, open('mnl_res.dump', 'w'))\nmnl_res_l = pickle.load(open('mnl_res.dump', 'r'))\n"