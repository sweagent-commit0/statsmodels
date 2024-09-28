"""First 100 days of the US House of Representatives 1995"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission from the original author,\nwho retains all rights.'
TITLE = __doc__
SOURCE = "\nJeff Gill's `Generalized Linear Models: A Unifited Approach`\n\nhttp://jgill.wustl.edu/research/books.html\n"
DESCRSHORT = 'Number of bill assignments in the 104th House in 1995'
DESCRLONG = "The example in Gill, seeks to explain the number of bill\nassignments in the first 100 days of the US' 104th House of Representatives.\nThe response variable is the number of bill assignments in the first 100 days\nover 20 Committees.  The explanatory variables in the example are the number of\nassignments in the first 100 days of the 103rd House, the number of members on\nthe committee, the number of subcommittees, the log of the number of staff\nassigned to the committee, a dummy variable indicating whether\nthe committee is a high prestige committee, and an interaction term between\nthe number of subcommittees and the log of the staff size.\n\nThe data returned by load are not cleaned to represent the above example.\n"
NOTE = '::\n\n    Number of Observations - 20\n    Number of Variables - 6\n    Variable name definitions::\n\n        BILLS104 - Number of bill assignments in the first 100 days of the\n                   104th House of Representatives.\n        SIZE     - Number of members on the committee.\n        SUBS     - Number of subcommittees.\n        STAFF    - Number of staff members assigned to the committee.\n        PRESTIGE - PRESTIGE == 1 is a high prestige committee.\n        BILLS103 - Number of bill assignments in the first 100 days of the\n                   103rd House of Representatives.\n\n    Committee names are included as a variable in the data file though not\n    returned by load.\n'

def load():
    """Load the committee data and returns a data class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass