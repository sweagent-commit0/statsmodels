"""Danish Money Demand Data"""
import pandas as pd
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = '\nDanish data used in S. Johansen and K. Juselius.  For estimating\nestimating a money demand function::\n\n    [1] Johansen, S. and Juselius, K. (1990), Maximum Likelihood Estimation\n        and Inference on Cointegration - with Applications to the Demand\n        for Money, Oxford Bulletin of Economics and Statistics, 52, 2,\n        169-210.\n'
DESCRSHORT = 'Danish Money Demand Data'
DESCRLONG = DESCRSHORT
NOTE = '::\n    Number of Observations - 55\n\n    Number of Variables - 5\n\n    Variable name definitions::\n\n        lrm - Log real money\n        lry - Log real income\n        lpy - Log prices\n        ibo - Bond rate\n        ide - Deposit rate\n'

def load():
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The Dataset instance does not contain endog and exog attributes.
    """
    pass
variable_names = ['lrm', 'lry', 'lpy', 'ibo', 'ide']

def __str__():
    return 'danish_data'