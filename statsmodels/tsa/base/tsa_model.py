from __future__ import annotations
from statsmodels.compat.pandas import is_float_index, is_int_index, is_numeric_dtype
import numbers
import warnings
import numpy as np
from pandas import DatetimeIndex, Index, Period, PeriodIndex, RangeIndex, Series, Timestamp, date_range, period_range, to_datetime
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
_tsa_doc = "\n    %(model)s\n\n    Parameters\n    ----------\n    %(params)s\n    dates : array_like, optional\n        An array-like object of datetime objects. If a pandas object is given\n        for endog or exog, it is assumed to have a DateIndex.\n    freq : str, optional\n        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',\n        'M', 'A', or 'Q'. This is optional if dates are given.\n    %(extra_params)s\n    %(extra_sections)s"
_model_doc = 'Timeseries model base class'
_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc

def get_index_loc(key, index):
    """
    Get the location of a specific key in an index

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or a location if the underlying index is a RangeIndex
        or an Index with an integer dtype.
    index : pd.Index
        The index to search.

    Returns
    -------
    loc : int
        The location of the key
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    If `key` is past the end of of the given index, and the index is either
    an Index with an integral dtype or a date index, this function extends
    the index up to and including key, and then returns the location in the
    new index.
    """
    pass

def get_index_label_loc(key, index, row_labels):
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or is only being used as row labels, or a location if
        the underlying index is a RangeIndex or a NumericIndex.
    index : pd.Index
        The index to search.
    row_labels : pd.Index
        Row labels to search if key not found in index

    Returns
    -------
    loc : int
        The location of the key
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    This function expands on `get_index_loc` by first trying the given
    base index (or the model's index if the base index was not given) and
    then falling back to try again with the model row labels as the base
    index.
    """
    pass

def get_prediction_index(start, end, nobs, base_index, index=None, silent=False, index_none=False, index_generated=None, data=None) -> tuple[int, int, int, Index | None]:
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    start : label
        The key at which to start prediction. Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    end : label
        The key at which to end prediction (note that this key will be
        *included* in prediction). Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    nobs : int
    base_index : pd.Index

    index : pd.Index, optional
        Optionally an index to associate the predicted results to. If None,
        an attempt is made to create an index for the predicted results
        from the model's index or model's row labels.
    silent : bool, optional
        Argument to silence warnings.

    Returns
    -------
    start : int
        The index / observation location at which to begin prediction.
    end : int
        The index / observation location at which to end in-sample
        prediction. The maximum value for this is nobs-1.
    out_of_sample : int
        The number of observations to forecast after the end of the sample.
    prediction_index : pd.Index or None
        The index associated with the prediction results. This index covers
        the range [start, end + out_of_sample]. If the model has no given
        index and no given row labels (i.e. endog/exog is not Pandas), then
        this will be None.

    Notes
    -----
    The arguments `start` and `end` behave differently, depending on if
    they are integer or not. If either is an integer, then it is assumed
    to refer to a *location* in the index, not to an index value. On the
    other hand, if it is a date string or some other type of object, then
    it is assumed to refer to an index *value*. In all cases, the returned
    `start` and `end` values refer to index *locations* (so in the former
    case, the given location is validated and returned whereas in the
    latter case a location is found that corresponds to the given index
    value).

    This difference in behavior is necessary to support `RangeIndex`. This
    is because integers for a RangeIndex could refer either to index values
    or to index locations in an ambiguous way (while for `NumericIndex`,
    since we have required them to be full indexes, there is no ambiguity).
    """
    pass

class TimeSeriesModel(base.LikelihoodModel):
    __doc__ = _tsa_doc % {'model': _model_doc, 'params': _generic_params, 'extra_params': _missing_param_doc, 'extra_sections': ''}

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none', **kwargs):
        super().__init__(endog, exog, missing=missing, **kwargs)
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
        """
        Initialize dates

        Parameters
        ----------
        dates : array_like, optional
            An array like object containing dates.
        freq : str, tuple, datetime.timedelta, DateOffset or None, optional
            A frequency specification for either `dates` or the row labels from
            the endog / exog data.

        Notes
        -----
        Creates `self._index` and related attributes. `self._index` is always
        a Pandas index, and it is always NumericIndex, DatetimeIndex, or
        PeriodIndex.

        If Pandas objects, endog / exog may have any type of index. If it is
        an NumericIndex with values 0, 1, ..., nobs-1 or if it is (coerceable to)
        a DatetimeIndex or PeriodIndex *with an associated frequency*, then it
        is called a "supported" index. Otherwise it is called an "unsupported"
        index.

        Supported indexes are standardized (i.e. a list of date strings is
        converted to a DatetimeIndex) and the result is put in `self._index`.

        Unsupported indexes are ignored, and a supported NumericIndex is
        generated and put in `self._index`. Warnings are issued in this case
        to alert the user if the returned index from some operation (e.g.
        forecasting) is different from the original data's index. However,
        whenever possible (e.g. purely in-sample prediction), the original
        index is returned.

        The benefit of supported indexes is that they allow *forecasting*, i.e.
        it is possible to extend them in a reasonable way. Thus every model
        must have an underlying supported index, even if it is just a generated
        NumericIndex.
        """
        pass

    def _get_index_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or a location if the underlying index is a RangeIndex
            or an NumericIndex.
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        If `key` is past the end of of the given index, and the index is either
        an NumericIndex or a date index, this function extends the index up to
        and including key, and then returns the location in the new index.
        """
        pass

    def _get_index_label_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or is only being used as row labels, or a location if
            the underlying index is a RangeIndex or an NumericIndex.
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the given
        base index (or the model's index if the base index was not given) and
        then falling back to try again with the model row labels as the base
        index.
        """
        pass

    def _get_prediction_index(self, start, end, index=None, silent=False) -> tuple[int, int, int, Index | None]:
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        start : label
            The key at which to start prediction. Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        end : label
            The key at which to end prediction (note that this key will be
            *included* in prediction). Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        silent : bool, optional
            Argument to silence warnings.

        Returns
        -------
        start : int
            The index / observation location at which to begin prediction.
        end : int
            The index / observation location at which to end in-sample
            prediction. The maximum value for this is nobs-1.
        out_of_sample : int
            The number of observations to forecast after the end of the sample.
        prediction_index : pd.Index or None
            The index associated with the prediction results. This index covers
            the range [start, end + out_of_sample]. If the model has no given
            index and no given row labels (i.e. endog/exog is not Pandas), then
            this will be None.

        Notes
        -----
        The arguments `start` and `end` behave differently, depending on if
        they are integer or not. If either is an integer, then it is assumed
        to refer to a *location* in the index, not to an index value. On the
        other hand, if it is a date string or some other type of object, then
        it is assumed to refer to an index *value*. In all cases, the returned
        `start` and `end` values refer to index *locations* (so in the former
        case, the given location is validated and returned whereas in the
        latter case a location is found that corresponds to the given index
        value).

        This difference in behavior is necessary to support `RangeIndex`. This
        is because integers for a RangeIndex could refer either to index values
        or to index locations in an ambiguous way (while for `NumericIndex`,
        since we have required them to be full indexes, there is no ambiguity).
        """
        pass
    exog_names = property(_get_exog_names, _set_exog_names, None, 'The names of the exogenous variables.')

class TimeSeriesModelResults(base.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params, scale=1.0):
        self.data = model.data
        super().__init__(model, params, normalized_cov_params, scale)

class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates'}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(TimeSeriesResultsWrapper, TimeSeriesModelResults)