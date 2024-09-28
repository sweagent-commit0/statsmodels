"""Nile River Flows."""
import pandas as pd
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = 'Nile River flows at Ashwan 1871-1970'
SOURCE = '\nThis data is first analyzed in:\n\n    Cobb, G. W. 1978. "The Problem of the Nile: Conditional Solution to a\n        Changepoint Problem." *Biometrika*. 65.2, 243-51.\n'
DESCRSHORT = 'This dataset contains measurements on the annual flow of\nthe Nile as measured at Ashwan for 100 years from 1871-1970.'
DESCRLONG = DESCRSHORT + ' There is an apparent changepoint near 1898.'
NOTE = '::\n\n    Number of observations: 100\n    Number of variables: 2\n    Variable name definitions:\n\n        year - the year of the observations\n        volumne - the discharge at Aswan in 10^8, m^3\n'

def load():
    """
    Load the Nile data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass