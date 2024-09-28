"""Smoking and lung cancer in eight cities in China."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Intern. J. Epidemiol. (1992)'
TITLE = __doc__
SOURCE = '\nTranscribed from Z. Liu, Smoking and Lung Cancer Incidence in China,\nIntern. J. Epidemiol., 21:197-201, (1992).\n'
DESCRSHORT = 'Co-occurrence of lung cancer and smoking in 8 Chinese cities.'
DESCRLONG = 'This is a series of 8 2x2 contingency tables showing the co-occurrence\nof lung cancer and smoking in 8 Chinese cities.\n'
NOTE = "::\n\n    Number of Observations - 8\n    Number of Variables - 3\n    Variable name definitions::\n\n        city_name - name of the city\n        smoking - yes or no, according to a person's smoking behavior\n        lung_cancer - yes or no, according to a person's lung cancer status\n"

def load_pandas():
    """
    Load the China smoking/lung cancer data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load():
    """
    Load the China smoking/lung cancer data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass