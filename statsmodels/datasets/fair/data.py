"""Fair's Extramarital Affairs Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Included with permission of the author.'
TITLE = 'Affairs dataset'
SOURCE = '\nFair, Ray. 1978. "A Theory of Extramarital Affairs," `Journal of Political\nEconomy`, February, 45-61.\n\nThe data is available at http://fairmodel.econ.yale.edu/rayfair/pdf/2011b.htm\n'
DESCRSHORT = 'Extramarital affair data.'
DESCRLONG = "Extramarital affair data used to explain the allocation\nof an individual's time among work, time spent with a spouse, and time\nspent with a paramour. The data is used as an example of regression\nwith censored data."
NOTE = "::\n\n    Number of observations: 6366\n    Number of variables: 9\n    Variable name definitions:\n\n        rate_marriage   : How rate marriage, 1 = very poor, 2 = poor, 3 = fair,\n                        4 = good, 5 = very good\n        age             : Age\n        yrs_married     : No. years married. Interval approximations. See\n                        original paper for detailed explanation.\n        children        : No. children\n        religious       : How relgious, 1 = not, 2 = mildly, 3 = fairly,\n                        4 = strongly\n        educ            : Level of education, 9 = grade school, 12 = high\n                        school, 14 = some college, 16 = college graduate,\n                        17 = some graduate school, 20 = advanced degree\n        occupation      : 1 = student, 2 = farming, agriculture; semi-skilled,\n                        or unskilled worker; 3 = white-colloar; 4 = teacher\n                        counselor social worker, nurse; artist, writers;\n                        technician, skilled worker, 5 = managerial,\n                        administrative, business, 6 = professional with\n                        advanced degree\n        occupation_husb : Husband's occupation. Same as occupation.\n        affairs         : measure of time spent in extramarital affairs\n\n    See the original paper for more details.\n"

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass