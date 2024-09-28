"""Taxation Powers Vote for the Scottish Parliament 1997 dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission from the original author,\nwho retains all rights.'
TITLE = 'Taxation Powers Vote for the Scottish Parliament 1997'
SOURCE = "\nJeff Gill's `Generalized Linear Models: A Unified Approach`\n\nhttp://jgill.wustl.edu/research/books.html\n"
DESCRSHORT = "Taxation Powers' Yes Vote for Scottish Parliamanet-1997"
DESCRLONG = "\nThis data is based on the example in Gill and describes the proportion of\nvoters who voted Yes to grant the Scottish Parliament taxation powers.\nThe data are divided into 32 council districts.  This example's explanatory\nvariables include the amount of council tax collected in pounds sterling as\nof April 1997 per two adults before adjustments, the female percentage of\ntotal claims for unemployment benefits as of January, 1998, the standardized\nmortality rate (UK is 100), the percentage of labor force participation,\nregional GDP, the percentage of children aged 5 to 15, and an interaction term\nbetween female unemployment and the council tax.\n\nThe original source files and variable information are included in\n/scotland/src/\n"
NOTE = "::\n\n    Number of Observations - 32 (1 for each Scottish district)\n\n    Number of Variables - 8\n\n    Variable name definitions::\n\n        YES    - Proportion voting yes to granting taxation powers to the\n                 Scottish parliament.\n        COUTAX - Amount of council tax collected in pounds steling as of\n                 April '97\n        UNEMPF - Female percentage of total unemployment benefits claims as of\n                January 1998\n        MOR    - The standardized mortality rate (UK is 100)\n        ACT    - Labor force participation (Short for active)\n        GDP    - GDP per county\n        AGE    - Percentage of children aged 5 to 15 in the county\n        COUTAX_FEMALEUNEMP - Interaction between COUTAX and UNEMPF\n\n    Council district names are included in the data file, though are not\n    returned by load.\n"

def load():
    """
    Load the Scotvote data and returns a Dataset instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the Scotvote data and returns a Dataset instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass