"""
Base tools for handling various kinds of data structures, attaching metadata to
results, and doing data cleaning
"""
from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError

def _asarray_2d_null_rows(x):
    """
    Makes sure input is an array and is 2d. Makes sure output is 2d. True
    indicates a null in the rows of 2d x.
    """
    pass

def _nan_rows(*arrs):
    """
    Returns a boolean array which is True where any of the rows in any
    of the _2d_ arrays in arrs are NaNs. Inputs can be any mixture of Series,
    DataFrames or array_like.
    """
    pass

class ModelData:
    """
    Class responsible for handling input data and extracting metadata into the
    appropriate form
    """
    _param_names = None
    _cov_names = None

    def __init__(self, endog, exog=None, missing='none', hasconst=None, **kwargs):
        if data_util._is_recarray(endog) or data_util._is_recarray(exog):
            from statsmodels.tools.sm_exceptions import recarray_exception
            raise NotImplementedError(recarray_exception)
        if 'design_info' in kwargs:
            self.design_info = kwargs.pop('design_info')
        if 'formula' in kwargs:
            self.formula = kwargs.pop('formula')
        if missing != 'none':
            arrays, nan_idx = self.handle_missing(endog, exog, missing, **kwargs)
            self.missing_row_idx = nan_idx
            self.__dict__.update(arrays)
            self.orig_endog = self.endog
            self.orig_exog = self.exog
            self.endog, self.exog = self._convert_endog_exog(self.endog, self.exog)
        else:
            self.__dict__.update(kwargs)
            self.orig_endog = endog
            self.orig_exog = exog
            self.endog, self.exog = self._convert_endog_exog(endog, exog)
        self.const_idx = None
        self.k_constant = 0
        self._handle_constant(hasconst)
        self._check_integrity()
        self._cache = {}

    def __getstate__(self):
        from copy import copy
        d = copy(self.__dict__)
        if 'design_info' in d:
            del d['design_info']
            d['restore_design_info'] = True
        return d

    def __setstate__(self, d):
        if 'restore_design_info' in d:
            from patsy import dmatrices, PatsyError
            exc = []
            try:
                data = d['frame']
            except KeyError:
                data = d['orig_endog'].join(d['orig_exog'])
            for depth in [2, 3, 1, 0, 4]:
                try:
                    _, design = dmatrices(d['formula'], data, eval_env=depth, return_type='dataframe')
                    break
                except (NameError, PatsyError) as e:
                    exc.append(e)
                    pass
            else:
                raise exc[-1]
            self.design_info = design.design_info
            del d['restore_design_info']
        self.__dict__.update(d)

    @classmethod
    def handle_missing(cls, endog, exog, missing, **kwargs):
        """
        This returns a dictionary with keys endog, exog and the keys of
        kwargs. It preserves Nones.
        """
        pass

    @property
    def cov_names(self):
        """
        Labels for covariance matrices

        In multidimensional models, each dimension of a covariance matrix
        differs from the number of param_names.

        If not set, returns param_names
        """
        pass

class PatsyData(ModelData):
    pass

class PandasData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """

def handle_data_class_factory(endog, exog):
    """
    Given inputs
    """
    pass