"""
Look at some macro plots, then do some VARs and IRFs.
"""
import numpy as np
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tplt
import statsmodels.api as sm
data = sm.datasets.macrodata.load()
data = data.data
dates = ts.date_array(start_date=ts.Date('Q', year=1959, quarter=1), end_date=ts.Date('Q', year=2009, quarter=3))
ts_data = data[['realgdp', 'realcons', 'cpi']].view(float).reshape(-1, 3)
ts_data = np.column_stack((ts_data, (1 - data['unemp'] / 100) * data['pop']))
ts_series = ts.time_series(ts_data, dates)
fig = tplt.tsfigure()
fsp = fig.add_tsplot(221)
fsp.tsplot(ts_series[:, 0], '-')
fsp.set_title('Real GDP')
fsp = fig.add_tsplot(222)
fsp.tsplot(ts_series[:, 1], 'r-')
fsp.set_title('Real Consumption')
fsp = fig.add_tsplot(223)
fsp.tsplot(ts_series[:, 2], 'g-')
fsp.set_title('CPI')
fsp = fig.add_tsplot(224)
fsp.tsplot(ts_series[:, 3], 'y-')
fsp.set_title('Employment')