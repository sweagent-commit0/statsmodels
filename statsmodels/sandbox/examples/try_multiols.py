"""

Created on Sun May 26 13:23:40 2013

Author: Josef Perktold, based on Enrico Giampieri's multiOLS
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.multilinear import multiOLS, multigroup
data = sm.datasets.longley.load_pandas()
df = data.exog
df['TOTEMP'] = data.endog
res0 = multiOLS('GNP + 1', df)
res = multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])
print(res.to_string())
url = 'https://raw.githubusercontent.com/vincentarelbundock/'
url = url + 'Rdatasets/csv/HistData/Guerry.csv'
df = pd.read_csv(url, index_col=1)
pvals = multiOLS('Wealth', df)['adj_pvals', '_f_test']
groups = {}
groups['crime'] = ['Crime_prop', 'Infanticide', 'Crime_parents', 'Desertion', 'Crime_pers']
groups['religion'] = ['Donation_clergy', 'Clergy', 'Donations']
groups['wealth'] = ['Commerce', 'Lottery', 'Instruction', 'Literacy']
res3 = multigroup(pvals < 0.05, groups)
print(res3)