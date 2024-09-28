"""US Capital Punishment dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission from the original author,\nwho retains all rights.'
TITLE = __doc__
SOURCE = "\nJeff Gill's `Generalized Linear Models: A Unified Approach`\n\nhttp://jgill.wustl.edu/research/books.html\n"
DESCRSHORT = 'Number of state executions in 1997'
DESCRLONG = 'This data describes the number of times capital punishment is implemented\nat the state level for the year 1997.  The outcome variable is the number of\nexecutions.  There were executions in 17 states.\nIncluded in the data are explanatory variables for median per capita income\nin dollars, the percent of the population classified as living in poverty,\nthe percent of Black citizens in the population, the rate of violent\ncrimes per 100,000 residents for 1996, a dummy variable indicating\nwhether the state is in the South, and (an estimate of) the proportion\nof the population with a college degree of some kind.\n'
NOTE = '::\n\n    Number of Observations - 17\n    Number of Variables - 7\n    Variable name definitions::\n\n        EXECUTIONS - Executions in 1996\n        INCOME - Median per capita income in 1996 dollars\n        PERPOVERTY - Percent of the population classified as living in poverty\n        PERBLACK - Percent of black citizens in the population\n        VC100k96 - Rate of violent crimes per 100,00 residents for 1996\n        SOUTH - SOUTH == 1 indicates a state in the South\n        DEGREE - An esimate of the proportion of the state population with a\n            college degree of some kind\n\n    State names are included in the data file, though not returned by load.\n'

def load_pandas():
    """
    Load the cpunish data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load():
    """
    Load the cpunish data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass