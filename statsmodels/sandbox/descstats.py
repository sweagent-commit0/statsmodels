"""
Glue for returning descriptive statistics.
"""
import os
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

def descstats(data, cols=None, axis=0):
    """
    Prints descriptive statistics for one or multiple variables.

    Parameters
    ----------
    data: numpy array
        `x` is the data

    v: list, optional
        A list of the column number of variables.
        Default is all columns.

    axis: 1 or 0
        axis order of data.  Default is 0 for column-ordered data.

    Examples
    --------
    >>> descstats(data.exog,v=['x_1','x_2','x_3'])
    """
    pass
if __name__ == '__main__':
    import statsmodels.api as sm
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    sum1 = descstats(data.exog)
    sum1a = descstats(data.exog[:, :1])
    if os.path.isfile('./Econ724_PS_I_Data.csv'):
        data2 = np.recfromcsv('./Econ724_PS_I_Data.csv')
        sum2 = descstats(data2.ahe)
        sum3 = descstats(np.column_stack((data2.ahe, data2.yrseduc)))
        sum4 = descstats(np.column_stack([data2[_] for _ in data2.dtype.names]))