"""Grunfeld (1950) Investment Data"""
import pandas as pd
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = 'This is the Grunfeld (1950) Investment Data.\n\nThe source for the data was the original 11-firm data set from Grunfeld\'s Ph.D.\nthesis recreated by Kleiber and Zeileis (2008) "The Grunfeld Data at 50".\nThe data can be found here.\nhttp://statmath.wu-wien.ac.at/~zeileis/grunfeld/\n\nFor a note on the many versions of the Grunfeld data circulating see:\nhttp://www.stanford.edu/~clint/bench/grunfeld.htm\n'
DESCRSHORT = 'Grunfeld (1950) Investment Data for 11 U.S. Firms.'
DESCRLONG = DESCRSHORT
NOTE = '::\n\n    Number of observations - 220 (20 years for 11 firms)\n\n    Number of variables - 5\n\n    Variables name definitions::\n\n        invest  - Gross investment in 1947 dollars\n        value   - Market value as of Dec. 31 in 1947 dollars\n        capital - Stock of plant and equipment in 1947 dollars\n        firm    - General Motors, US Steel, General Electric, Chrysler,\n                Atlantic Refining, IBM, Union Oil, Westinghouse, Goodyear,\n                Diamond Match, American Steel\n        year    - 1935 - 1954\n\n    Note that raw_data has firm expanded to dummy variables, since it is a\n    string categorical variable.\n'

def load():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    pass

def load_pandas():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    pass