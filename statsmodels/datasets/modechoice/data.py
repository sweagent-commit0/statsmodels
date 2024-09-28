"""Travel Mode Choice"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = "\nGreene, W.H. and D. Hensher (1997) Multinomial logit and discrete choice models\nin Greene, W. H. (1997) LIMDEP version 7.0 user's manual revised, Plainview,\nNew York econometric software, Inc.\nDownload from on-line complements to Greene, W.H. (2011) Econometric Analysis,\nPrentice Hall, 7th Edition (data table F18-2)\nhttp://people.stern.nyu.edu/wgreene/Text/Edition7/TableF18-2.csv\n"
DESCRSHORT = 'Data used to study travel mode choice between Australian cities\n'
DESCRLONG = 'The data, collected as part of a 1987 intercity mode choice\nstudy, are a sub-sample of 210 non-business trips between Sydney, Canberra and\nMelbourne in which the traveler chooses a mode from four alternatives (plane,\ncar, bus and train). The sample, 840 observations, is choice based with\nover-sampling of the less popular modes (plane, train and bus) and under-sampling\nof the more popular mode, car. The level of service data was derived from highway\nand transport networks in Sydney, Melbourne, non-metropolitan N.S.W. and Victoria,\nincluding the Australian Capital Territory.'
NOTE = '::\n\n    Number of observations: 840 Observations On 4 Modes for 210 Individuals.\n    Number of variables: 8\n    Variable name definitions::\n\n        individual = 1 to 210\n        mode =\n            1 - air\n            2 - train\n            3 - bus\n            4 - car\n        choice =\n            0 - no\n            1 - yes\n        ttme = terminal waiting time for plane, train and bus (minutes); 0\n               for car.\n        invc = in vehicle cost for all stages (dollars).\n        invt = travel time (in-vehicle time) for all stages (minutes).\n        gc = generalized cost measure:invc+(invt*value of travel time savings)\n            (dollars).\n        hinc = household income ($1000s).\n        psize = traveling group size in mode chosen (number).'

def load():
    """
    Load the data modechoice data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load_pandas():
    """
    Load the data modechoice data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass