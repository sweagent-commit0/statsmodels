"""Name of dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = 'Engel (1857) food expenditure data'
SOURCE = '\nThis dataset was used in Koenker and Bassett (1982) and distributed alongside\nthe ``quantreg`` package for R.\n\nKoenker, R. and Bassett, G (1982) Robust Tests of Heteroscedasticity based on\nRegression Quantiles; Econometrica 50, 43-61.\n\nRoger Koenker (2012). quantreg: Quantile Regression. R package version 4.94.\nhttp://CRAN.R-project.org/package=quantreg\n'
DESCRSHORT = 'Engel food expenditure data.'
DESCRLONG = 'Data on income and food expenditure for 235 working class households in 1857 Belgium.'
NOTE = '::\n\n    Number of observations: 235\n    Number of variables: 2\n    Variable name definitions:\n        income - annual household income (Belgian francs)\n        foodexp - annual household food expenditure (Belgian francs)\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass