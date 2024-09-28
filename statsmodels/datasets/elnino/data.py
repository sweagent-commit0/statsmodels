"""El Nino dataset, 1950 - 2010"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This data is in the public domain.'
TITLE = 'El Nino - Sea Surface Temperatures'
SOURCE = "\nNational Oceanic and Atmospheric Administration's National Weather Service\n\nERSST.V3B dataset, Nino 1+2\nhttp://www.cpc.ncep.noaa.gov/data/indices/\n"
DESCRSHORT = 'Averaged monthly sea surface temperature - Pacific Ocean.'
DESCRLONG = 'This data contains the averaged monthly sea surface\ntemperature in degrees Celcius of the Pacific Ocean, between 0-10 degrees South\nand 90-80 degrees West, from 1950 to 2010.  This dataset was obtained from\nNOAA.\n'
NOTE = '::\n\n    Number of Observations - 61 x 12\n\n    Number of Variables - 1\n\n    Variable name definitions::\n\n        TEMPERATURE - average sea surface temperature in degrees Celcius\n                      (12 columns, one per month).\n'

def load():
    """
    Load the El Nino data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The elnino Dataset instance does not contain endog and exog attributes.
    """
    pass