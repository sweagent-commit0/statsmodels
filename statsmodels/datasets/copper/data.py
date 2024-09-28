"""World Copper Prices 1951-1975 dataset."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Used with express permission from the original author,\nwho retains all rights.'
TITLE = 'World Copper Market 1951-1975 Dataset'
SOURCE = "\nJeff Gill's `Generalized Linear Models: A Unified Approach`\n\nhttp://jgill.wustl.edu/research/books.html\n"
DESCRSHORT = 'World Copper Market 1951-1975'
DESCRLONG = 'This data describes the world copper market from 1951 through 1975.  In an\nexample, in Gill, the outcome variable (of a 2 stage estimation) is the world\nconsumption of copper for the 25 years.  The explanatory variables are the\nworld consumption of copper in 1000 metric tons, the constant dollar adjusted\nprice of copper, the price of a substitute, aluminum, an index of real per\ncapita income base 1970, an annual measure of manufacturer inventory change,\nand a time trend.\n'
NOTE = '\nNumber of Observations - 25\n\nNumber of Variables - 6\n\nVariable name definitions::\n\n    WORLDCONSUMPTION - World consumption of copper (in 1000 metric tons)\n    COPPERPRICE - Constant dollar adjusted price of copper\n    INCOMEINDEX - An index of real per capita income (base 1970)\n    ALUMPRICE - The price of aluminum\n    INVENTORYINDEX - A measure of annual manufacturer inventory trend\n    TIME - A time trend\n\nYears are included in the data file though not returned by load.\n'

def load_pandas():
    """
    Load the copper data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load():
    """
    Load the copper data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass