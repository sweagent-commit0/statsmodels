"""Star98 Educational Testing dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission from the original author,\nwho retains all rights.'
TITLE = 'Star98 Educational Dataset'
SOURCE = "\nJeff Gill's `Generalized Linear Models: A Unified Approach`\n\nhttp://jgill.wustl.edu/research/books.html\n"
DESCRSHORT = 'Math scores for 303 student with 10 explanatory factors'
DESCRLONG = '\nThis data is on the California education policy and outcomes (STAR program\nresults for 1998.  The data measured standardized testing by the California\nDepartment of Education that required evaluation of 2nd - 11th grade students\nby the the Stanford 9 test on a variety of subjects.  This dataset is at\nthe level of the unified school district and consists of 303 cases.  The\nbinary response variable represents the number of 9th graders scoring\nover the national median value on the mathematics exam.\n\nThe data used in this example is only a subset of the original source.\n'
NOTE = "::\n\n    Number of Observations - 303 (counties in California).\n\n    Number of Variables - 13 and 8 interaction terms.\n\n    Definition of variables names::\n\n        NABOVE   - Total number of students above the national median for the\n                   math section.\n        NBELOW   - Total number of students below the national median for the\n                   math section.\n        LOWINC   - Percentage of low income students\n        PERASIAN - Percentage of Asian student\n        PERBLACK - Percentage of black students\n        PERHISP  - Percentage of Hispanic students\n        PERMINTE - Percentage of minority teachers\n        AVYRSEXP - Sum of teachers' years in educational service divided by the\n                number of teachers.\n        AVSALK   - Total salary budget including benefits divided by the number\n                   of full-time teachers (in thousands)\n        PERSPENK - Per-pupil spending (in thousands)\n        PTRATIO  - Pupil-teacher ratio.\n        PCTAF    - Percentage of students taking UC/CSU prep courses\n        PCTCHRT  - Percentage of charter schools\n        PCTYRRND - Percentage of year-round schools\n\n        The below variables are interaction terms of the variables defined\n        above.\n\n        PERMINTE_AVYRSEXP\n        PEMINTE_AVSAL\n        AVYRSEXP_AVSAL\n        PERSPEN_PTRATIO\n        PERSPEN_PCTAF\n        PTRATIO_PCTAF\n        PERMINTE_AVTRSEXP_AVSAL\n        PERSPEN_PTRATIO_PCTAF\n"

def load():
    """
    Load the star98 data and returns a Dataset class instance.

    Returns
    -------
    Load instance:
        a class of the data with array attrbutes 'endog' and 'exog'
    """
    pass