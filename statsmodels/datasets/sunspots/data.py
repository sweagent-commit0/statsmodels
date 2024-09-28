"""Yearly sunspots data 1700-2008"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This data is public domain.'
TITLE = __doc__
SOURCE = '\nhttp://www.ngdc.noaa.gov/stp/solar/solarda3.html\n\nThe original dataset contains monthly data on sunspot activity in the file\n./src/sunspots_yearly.dat.  There is also sunspots_monthly.dat.\n'
DESCRSHORT = 'Yearly (1700-2008) data on sunspots from the National\nGeophysical Data Center.'
DESCRLONG = DESCRSHORT
NOTE = "::\n\n    Number of Observations - 309 (Annual 1700 - 2008)\n    Number of Variables - 1\n    Variable name definitions::\n\n        SUNACTIVITY - Number of sunspots for each year\n\n    The data file contains a 'YEAR' variable that is not returned by load.\n"

def load():
    """
    Load the yearly sunspot data and returns a data class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    This dataset only contains data for one variable, so the attributes
    data, raw_data, and endog are all the same variable.  There is no exog
    attribute defined.
    """
    pass