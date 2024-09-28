"""Heart Transplant Data, Miller 1976"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = '???'
TITLE = 'Transplant Survival Data'
SOURCE = 'Miller, R. (1976). Least squares regression with censored data. Biometrica, 63 (3). 449-464.\n\n'
DESCRSHORT = 'Survival times after receiving a heart transplant'
DESCRLONG = 'This data contains the survival time after receiving a heart transplant, the age of the patient and whether or not the survival time was censored.\n'
NOTE = '::\n\n    Number of Observations - 69\n\n    Number of Variables - 3\n\n    Variable name definitions::\n        death - Days after surgery until death\n        age - age at the time of surgery\n        censored - indicates if an observation is censored.  1 is uncensored\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass