"""Mauna Loa Weekly Atmospheric CO2 Data"""
import pandas as pd
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = 'Mauna Loa Weekly Atmospheric CO2 Data'
SOURCE = '\nData obtained from http://cdiac.ornl.gov/trends/co2/sio-keel-flask/sio-keel-flaskmlo_c.html\n\nObtained on 3/15/2014.\n\nCitation:\n\nKeeling, C.D. and T.P. Whorf. 2004. Atmospheric CO2 concentrations derived from flask air samples at sites in the SIO network. In Trends: A Compendium of Data on Global Change. Carbon Dioxide Information Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy, Oak Ridge, Tennessee, U.S.A.\n'
DESCRSHORT = 'Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.'
DESCRLONG = '\nAtmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.\n\nPeriod of Record: March 1958 - December 2001\n\nMethods: An Applied Physics Corporation (APC) nondispersive infrared gas analyzer was used to obtain atmospheric CO2 concentrations, based on continuous data (four measurements per hour) from atop intake lines on several towers. Steady data periods of not less than six hours per day are required; if no such six-hour periods are available on any given day, then no data are used that day. Weekly averages were calculated for most weeks throughout the approximately 44 years of record. The continuous data for year 2000 is compared with flask data from the same site in the graphics section.'
NOTE = '::\n\n    Number of observations: 2225\n    Number of variables: 2\n    Variable name definitions:\n\n        date - sample date in YYMMDD format\n        co2 - CO2 Concentration ppmv\n\n    The data returned by load_pandas contains the dates as the index.\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass