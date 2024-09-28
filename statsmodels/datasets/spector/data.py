"""Spector and Mazzeo (1980) - Program Effectiveness Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission of the original author, who\nretains all rights. '
TITLE = __doc__
SOURCE = "\nhttp://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm\n\nThe raw data was downloaded from Bill Greene's Econometric Analysis web site,\nthough permission was obtained from the original researcher, Dr. Lee Spector,\nProfessor of Economics, Ball State University."
DESCRSHORT = 'Experimental data on the effectiveness of the personalized\nsystem of instruction (PSI) program'
DESCRLONG = DESCRSHORT
NOTE = "::\n\n    Number of Observations - 32\n\n    Number of Variables - 4\n\n    Variable name definitions::\n\n        Grade - binary variable indicating whether or not a student's grade\n                improved.  1 indicates an improvement.\n        TUCE  - Test score on economics test\n        PSI   - participation in program\n        GPA   - Student's grade point average\n"

def load():
    """
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass