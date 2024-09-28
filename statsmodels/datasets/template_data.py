"""Name of dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'E.g., This is public domain.'
TITLE = 'Title of the dataset'
SOURCE = "\nThis section should provide a link to the original dataset if possible and\nattribution and correspondance information for the dataset's original author\nif so desired.\n"
DESCRSHORT = 'A short description.'
DESCRLONG = 'A longer description of the dataset.'
NOTE = '\n::\n\n    Number of observations:\n    Number of variables:\n    Variable name definitions:\n\nAny other useful information that does not fit into the above categories.\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass