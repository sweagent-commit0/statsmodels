import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp', 'realcons', 'realinv']]
names = mdata.dtype.names
data = mdata.view((float, 3))
use_growthrate = False
if use_growthrate:
    data = 100 * 4 * np.diff(np.log(data), axis=0)
model = VAR(data, names=names)
res = model.fit(4)
nobs_all = data.shape[0]
fc_in = np.array([np.squeeze(res.forecast(model.y[t - 20:t], 1)) for t in range(nobs_all - 6, nobs_all)])
print(fc_in - res.fittedvalues[-6:])
fc_out = np.array([np.squeeze(VAR(data[:t]).fit(2).forecast(data[t - 20:t], 1)) for t in range(nobs_all - 6, nobs_all)])
print(fc_out - data[nobs_all - 6:nobs_all])
print(fc_out - res.fittedvalues[-6:])
h = 2
fc_out = np.array([VAR(data[:t]).fit(2).forecast(data[t - 20:t], h)[-1] for t in range(nobs_all - 6 - h + 1, nobs_all - h + 1)])
print(fc_out - data[nobs_all - 6:nobs_all])
print(fc_out - res.fittedvalues[-6:])
res.plot_forecast(20)