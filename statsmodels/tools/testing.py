"""assert functions from numpy and pandas testing

"""
from statsmodels.compat.pandas import testing as pdt
import numpy.testing as npt
import pandas
from statsmodels.tools.tools import Bunch
PARAM_LIST = ['params', 'bse', 'tvalues', 'pvalues']

def bunch_factory(attribute, columns):
    """
    Generates a special purpose Bunch class

    Parameters
    ----------
    attribute: str
        Attribute to access when splitting
    columns: List[str]
        List of names to use when splitting the columns of attribute

    Notes
    -----
    After the class is initialized as a Bunch, the columne of attribute
    are split so that Bunch has the keys in columns and
    bunch[column[i]] = bunch[attribute][:, i]
    """
    pass
ParamsTableTestBunch = bunch_factory('params_table', PARAM_LIST)
MarginTableTestBunch = bunch_factory('margins_table', PARAM_LIST)

class Holder:
    """
    Test-focused class to simplify accessing values by attribute
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        ss = '\n'.join((str(k) + ' = ' + str(v).replace('\n', '\n    ') for k, v in vars(self).items()))
        return ss

    def __repr__(self):
        ss = '\n'.join((str(k) + ' = ' + repr(v).replace('\n', '\n    ') for k, v in vars(self).items()))
        ss = str(self.__class__) + '\n' + ss
        return ss