"""Longley dataset"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = '\nThe classic 1967 Longley Data\n\nhttp://www.itl.nist.gov/div898/strd/lls/data/Longley.shtml\n\n::\n\n    Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the\n        Electronic Comptuer from the Point of View of the User."  Journal of\n        the American Statistical Association.  62.319, 819-41.\n'
DESCRSHORT = ''
DESCRLONG = 'The Longley dataset contains various US macroeconomic\nvariables that are known to be highly collinear.  It has been used to appraise\nthe accuracy of least squares routines.'
NOTE = '::\n\n    Number of Observations - 16\n\n    Number of Variables - 6\n\n    Variable name definitions::\n\n            TOTEMP - Total Employment\n            GNPDEFL - GNP deflator\n            GNP - GNP\n            UNEMP - Number of unemployed\n            ARMED - Size of armed forces\n            POP - Population\n            YEAR - Year (1947 - 1962)\n'

def load():
    """
    Load the Longley data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the Longley data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass