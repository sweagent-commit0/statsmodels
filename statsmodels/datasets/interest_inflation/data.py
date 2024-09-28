"""(West) German interest and inflation rate 1972-1998"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = '...'
TITLE = __doc__
SOURCE = '\nhttp://www.jmulti.de/download/datasets/e6.dat\n'
DESCRSHORT = '(West) German interest and inflation rate 1972Q2 - 1998Q4'
DESCRLONG = 'West German (until 1990) / German (afterwards) interest and\ninflation rate 1972Q2 - 1998Q4\n'
NOTE = '::\n    Number of Observations - 107\n\n    Number of Variables - 2\n\n    Variable name definitions::\n\n        year      - 1972q2 - 1998q4\n        quarter   - 1-4\n        Dp        - Delta log gdp deflator\n        R         - nominal long term interest rate\n'
variable_names = ['Dp', 'R']
first_season = 1

def load():
    """
    Load the West German interest/inflation data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The interest_inflation Dataset instance does not contain endog and exog
    attributes.
    """
    pass

def __str__():
    return 'e6'