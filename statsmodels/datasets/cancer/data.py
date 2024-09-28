"""Breast Cancer Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = '???'
TITLE = 'Breast Cancer Data'
SOURCE = "\nThis is the breast cancer data used in Owen's empirical likelihood.  It is taken from\nRice, J.A. Mathematical Statistics and Data Analysis.\nhttp://www.cengage.com/statistics/discipline_content/dataLibrary.html\n"
DESCRSHORT = 'Breast Cancer and county population'
DESCRLONG = 'The number of breast cancer observances in various counties'
NOTE = '::\n\n    Number of observations: 301\n    Number of variables: 2\n    Variable name definitions:\n\n        cancer - The number of breast cancer observances\n        population - The population of the county\n\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass