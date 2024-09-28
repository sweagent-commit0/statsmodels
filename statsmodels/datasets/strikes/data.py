"""U.S. Strike Duration Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = '\nThis is a subset of the data used in Kennan (1985). It was originally\npublished by the Bureau of Labor Statistics.\n\n::\n\n    Kennan, J. 1985. "The duration of contract strikes in US manufacturing.\n        `Journal of Econometrics` 28.1, 5-28.\n'
DESCRSHORT = 'Contains data on the length of strikes in US manufacturing and\nunanticipated industrial production.'
DESCRLONG = 'Contains data on the length of strikes in US manufacturing and\nunanticipated industrial production. The data is a subset of the data originally\nused by Kennan. The data here is data for the months of June only to avoid\nseasonal issues.'
NOTE = '::\n\n    Number of observations - 62\n\n    Number of variables - 2\n\n    Variable name definitions::\n\n                duration - duration of the strike in days\n                iprod - unanticipated industrial production\n'

def load_pandas():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass