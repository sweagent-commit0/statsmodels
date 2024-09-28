"""Stack loss data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain. '
TITLE = __doc__
SOURCE = '\nBrownlee, K. A. (1965), "Statistical Theory and Methodology in\nScience and Engineering", 2nd edition, New York:Wiley.\n'
DESCRSHORT = 'Stack loss plant data of Brownlee (1965)'
DESCRLONG = "The stack loss plant data of Brownlee (1965) contains\n21 days of measurements from a plant's oxidation of ammonia to nitric acid.\nThe nitric oxide pollutants are captured in an absorption tower."
NOTE = '::\n\n    Number of Observations - 21\n\n    Number of Variables - 4\n\n    Variable name definitions::\n\n        STACKLOSS - 10 times the percentage of ammonia going into the plant\n                    that escapes from the absoroption column\n        AIRFLOW   - Rate of operation of the plant\n        WATERTEMP - Cooling water temperature in the absorption tower\n        ACIDCONC  - Acid concentration of circulating acid minus 50 times 10.\n'

def load():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass