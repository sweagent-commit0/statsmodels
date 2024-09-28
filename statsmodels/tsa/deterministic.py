from statsmodels.compat.pandas import PD_LT_2_2_0, Appender, is_int_index, to_numpy
from abc import ABC, abstractmethod
import datetime as dt
from typing import Hashable, List, Optional, Sequence, Set, Tuple, Type, Union
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import bool_like, float_like, required_int_like, string_like
from statsmodels.tsa.tsatools import freq_to_period
DateLike = Union[dt.datetime, pd.Timestamp, np.datetime64]
IntLike = Union[int, np.integer]
START_BEFORE_INDEX_ERR = 'start is less than the first observation in the index. Values can only be created for observations after the start of the index.\n'

class DeterministicTerm(ABC):
    """Abstract Base Class for all Deterministic Terms"""
    _is_dummy = False

    @property
    def is_dummy(self) -> bool:
        """Flag indicating whether the values produced are dummy variables"""
        pass

    @abstractmethod
    def in_sample(self, index: Sequence[Hashable]) -> pd.DataFrame:
        """
        Produce deterministic trends for in-sample fitting.

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """
        pass

    @abstractmethod
    def out_of_sample(self, steps: int, index: Sequence[Hashable], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        """
        Produce deterministic trends for out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of steps to forecast
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.
        forecast_index : index_like
            An Index or index-like object to use for the forecasts. If
            provided must have steps elements.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """A meaningful string representation of the term"""

    def __hash__(self) -> int:
        name: Tuple[Hashable, ...] = (type(self).__name__,)
        return hash(name + self._eq_attr)

    @property
    @abstractmethod
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        """tuple of attributes that are used for equality comparison"""
        pass

    @staticmethod
    def _extend_index(index: pd.Index, steps: int, forecast_index: Optional[Sequence[Hashable]]=None) -> pd.Index:
        """Extend the forecast index"""
        pass

    def __repr__(self) -> str:
        return self.__str__() + f' at 0x{id(self):0x}'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            own_attr = self._eq_attr
            oth_attr = other._eq_attr
            if len(own_attr) != len(oth_attr):
                return False
            return all([a == b for a, b in zip(own_attr, oth_attr)])
        else:
            return False

class TimeTrendDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Time Trend Deterministic Terms"""

    def __init__(self, constant: bool=True, order: int=0) -> None:
        self._constant = bool_like(constant, 'constant')
        self._order = required_int_like(order, 'order')

    @property
    def constant(self) -> bool:
        """Flag indicating that a constant is included"""
        pass

    @property
    def order(self) -> int:
        """Order of the time trend"""
        pass

    def __str__(self) -> str:
        terms = []
        if self._constant:
            terms.append('Constant')
        if self._order:
            terms.append(f'Powers 1 to {self._order + 1}')
        if not terms:
            terms = ['Empty']
        terms_str = ','.join(terms)
        return f'TimeTrend({terms_str})'

class TimeTrend(TimeTrendDeterministicTerm):
    """
    Constant and time trend determinstic terms

    Parameters
    ----------
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).

    See Also
    --------
    DeterministicProcess
    Seasonality
    Fourier
    CalendarTimeTrend

    Examples
    --------
    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import TimeTrend
    >>> data = sunspots.load_pandas().data
    >>> trend_gen = TimeTrend(True, 3)
    >>> trend_gen.in_sample(data.index)
    """

    def __init__(self, constant: bool=True, order: int=0) -> None:
        super().__init__(constant, order)

    @classmethod
    def from_string(cls, trend: str) -> 'TimeTrend':
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend

        Returns
        -------
        TimeTrend
            The TimeTrend instance.
        """
        pass

class Seasonality(DeterministicTerm):
    """
    Seasonal dummy deterministic terms

    Parameters
    ----------
    period : int
        The length of a full cycle. Must be >= 2.
    initial_period : int
        The seasonal index of the first observation. 1-indexed so must
        be in {1, 2, ..., period}.

    See Also
    --------
    DeterministicProcess
    TimeTrend
    Fourier
    CalendarSeasonality

    Examples
    --------
    Solar data has an 11-year cycle

    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import Seasonality
    >>> data = sunspots.load_pandas().data
    >>> seas_gen = Seasonality(11)
    >>> seas_gen.in_sample(data.index)

    To start at a season other than 1

    >>> seas_gen = Seasonality(11, initial_period=4)
    >>> seas_gen.in_sample(data.index)
    """
    _is_dummy = True

    def __init__(self, period: int, initial_period: int=1) -> None:
        self._period = required_int_like(period, 'period')
        self._initial_period = required_int_like(initial_period, 'initial_period')
        if period < 2:
            raise ValueError('period must be >= 2')
        if not 1 <= self._initial_period <= period:
            raise ValueError('initial_period must be in {1, 2, ..., period}')

    @property
    def period(self) -> int:
        """The period of the seasonality"""
        pass

    @property
    def initial_period(self) -> int:
        """The seasonal index of the first observation"""
        pass

    @classmethod
    def from_index(cls, index: Union[Sequence[Hashable], pd.DatetimeIndex, pd.PeriodIndex]) -> 'Seasonality':
        """
        Construct a seasonality directly from an index using its frequency.

        Parameters
        ----------
        index : {DatetimeIndex, PeriodIndex}
            An index with its frequency (`freq`) set.

        Returns
        -------
        Seasonality
            The initialized Seasonality instance.
        """
        pass

    def __str__(self) -> str:
        return f'Seasonality(period={self._period})'

class FourierDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Fourier Deterministic Terms"""

    def __init__(self, order: int) -> None:
        self._order = required_int_like(order, 'terms')

    @property
    def order(self) -> int:
        """The order of the Fourier terms included"""
        pass

class Fourier(FourierDeterministicTerm):
    """
    Fourier series deterministic terms

    Parameters
    ----------
    period : int
        The length of a full cycle. Must be >= 2.
    order : int
        The number of Fourier components to include. Must be <= 2*period.

    See Also
    --------
    DeterministicProcess
    TimeTrend
    Seasonality
    CalendarFourier

    Notes
    -----
    Both a sine and a cosine term are included for each i=1, ..., order

    .. math::

       f_{i,s,t} & = \\sin\\left(2 \\pi i \\times \\frac{t}{m} \\right)  \\\\
       f_{i,c,t} & = \\cos\\left(2 \\pi i \\times \\frac{t}{m} \\right)

    where m is the length of the period.

    Examples
    --------
    Solar data has an 11-year cycle

    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import Fourier
    >>> data = sunspots.load_pandas().data
    >>> fourier_gen = Fourier(11, order=2)
    >>> fourier_gen.in_sample(data.index)
    """
    _is_dummy = False

    def __init__(self, period: float, order: int):
        super().__init__(order)
        self._period = float_like(period, 'period')
        if 2 * self._order > self._period:
            raise ValueError('2 * order must be <= period')

    @property
    def period(self) -> float:
        """The period of the Fourier terms"""
        pass

    def __str__(self) -> str:
        return f'Fourier(period={self._period}, order={self._order})'

class CalendarDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for calendar deterministic terms"""

    def __init__(self, freq: str) -> None:
        try:
            index = pd.date_range('2020-01-01', freq=freq, periods=1)
            self._freq = index.freq
        except ValueError:
            raise ValueError('freq is not understood by pandas')

    @property
    def freq(self) -> str:
        """The frequency of the deterministic terms"""
        pass

class CalendarFourier(CalendarDeterministicTerm, FourierDeterministicTerm):
    """
    Fourier series deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    order : int
        The number of Fourier components to include. Must be <= 2*period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarSeasonality
    Fourier

    Notes
    -----
    Both a sine and a cosine term are included for each i=1, ..., order

    .. math::

       f_{i,s,t} & = \\sin\\left(2 \\pi i \\tau_t \\right)  \\\\
       f_{i,c,t} & = \\cos\\left(2 \\pi i \\tau_t \\right)

    where m is the length of the period and :math:`\\tau_t` is the frequency
    normalized time.  For example, when freq is "D" then an observation with
    a timestamp of 12:00:00 would have :math:`\\tau_t=0.5`.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    Fourier terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarFourier
    >>> cal_fourier_gen = CalendarFourier("D", 2)
    >>> cal_fourier_gen.in_sample(index)
    """

    def __init__(self, freq: str, order: int) -> None:
        super().__init__(freq)
        FourierDeterministicTerm.__init__(self, order)
        self._order = required_int_like(order, 'terms')

    def __str__(self) -> str:
        return f'Fourier(freq={self._freq.freqstr}, order={self._order})'

class CalendarSeasonality(CalendarDeterministicTerm):
    """
    Seasonal dummy deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        The frequency of the seasonal effect.
    period : str
        The pandas frequency string describing the full period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarFourier
    Seasonality

    Examples
    --------
    Here we simulate irregularly spaced data (in time) and hourly seasonal
    dummies for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarSeasonality
    >>> cal_seas_gen = CalendarSeasonality("H", "D")
    >>> cal_seas_gen.in_sample(index)
    """
    _is_dummy = True
    if PD_LT_2_2_0:
        _supported = {'W': {'B': 5, 'D': 7, 'h': 24 * 7, 'H': 24 * 7}, 'D': {'h': 24, 'H': 24}, 'Q': {'MS': 3, 'M': 3}, 'A': {'MS': 12, 'M': 12}, 'Y': {'MS': 12, 'Q': 4, 'M': 12}}
    else:
        _supported = {'W': {'B': 5, 'D': 7, 'h': 24 * 7}, 'D': {'h': 24}, 'Q': {'MS': 3, 'ME': 3}, 'A': {'MS': 12, 'ME': 12, 'QE': 4}, 'Y': {'MS': 12, 'ME': 12, 'QE': 4}, 'QE': {'ME': 3}, 'YE': {'ME': 12, 'QE': 4}}

    def __init__(self, freq: str, period: str) -> None:
        freq_options: Set[str] = set()
        freq_options.update(*[list(val.keys()) for val in self._supported.values()])
        period_options = tuple(self._supported.keys())
        freq = string_like(freq, 'freq', options=tuple(freq_options), lower=False)
        period = string_like(period, 'period', options=period_options, lower=False)
        if freq not in self._supported[period]:
            raise ValueError(f'The combination of freq={freq} and period={period} is not supported.')
        super().__init__(freq)
        self._period = period
        self._freq_str = self._freq.freqstr.split('-')[0]

    @property
    def freq(self) -> str:
        """The frequency of the deterministic terms"""
        pass

    @property
    def period(self) -> str:
        """The full period"""
        pass

    def __str__(self) -> str:
        return f'Seasonal(freq={self._freq_str})'

class CalendarTimeTrend(CalendarDeterministicTerm, TimeTrendDeterministicTerm):
    """
    Constant and time trend determinstic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).
    base_period : {str, pd.Timestamp}, default None
        The base period to use when computing the time stamps. This value is
        treated as 1 and so all other time indices are defined as the number
        of periods since or before this time stamp. If not provided, defaults
        to pandas base period for a PeriodIndex.

    See Also
    --------
    DeterministicProcess
    CalendarFourier
    CalendarSeasonality
    TimeTrend

    Notes
    -----
    The time stamp, :math:`\\tau_t`, is the number of periods that have elapsed
    since the base_period. :math:`\\tau_t` may be fractional.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    time trend terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarTimeTrend
    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1)
    >>> cal_trend_gen.in_sample(index)

    Next, we normalize using the first time stamp

    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1,
    ...                                   base_period=index[0])
    >>> cal_trend_gen.in_sample(index)
    """

    def __init__(self, freq: str, constant: bool=True, order: int=0, *, base_period: Optional[Union[str, DateLike]]=None) -> None:
        super().__init__(freq)
        TimeTrendDeterministicTerm.__init__(self, constant=constant, order=order)
        self._ref_i8 = 0
        if base_period is not None:
            pr = pd.period_range(base_period, periods=1, freq=self._freq)
            self._ref_i8 = pr.asi8[0]
        self._base_period = None if base_period is None else str(base_period)

    @property
    def base_period(self) -> Optional[str]:
        """The base period"""
        pass

    @classmethod
    def from_string(cls, freq: str, trend: str, base_period: Optional[Union[str, DateLike]]=None) -> 'CalendarTimeTrend':
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        freq : str
            A string convertible to a pandas frequency.
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend
        base_period : {str, pd.Timestamp}, default None
            The base period to use when computing the time stamps. This value
            is treated as 1 and so all other time indices are defined as the
            number of periods since or before this time stamp. If not
            provided, defaults to pandas base period for a PeriodIndex.

        Returns
        -------
        TimeTrend
            The TimeTrend instance.
        """
        pass

    def __str__(self) -> str:
        value = TimeTrendDeterministicTerm.__str__(self)
        value = 'Calendar' + value[:-1] + f', freq={self._freq.freqstr})'
        if self._base_period is not None:
            value = value[:-1] + f'base_period={self._base_period})'
        return value

class DeterministicProcess:
    """
    Container class for deterministic terms.

    Directly supports constants, time trends, and either seasonal dummies or
    fourier terms for a single cycle. Additional deterministic terms beyond
    the set that can be directly initialized through the constructor can be
    added.

    Parameters
    ----------
    index : {Sequence[Hashable], pd.Index}
        The index of the process. Should usually be the "in-sample" index when
        used in forecasting applications.
    period : {float, int}, default None
        The period of the seasonal or fourier components. Must be an int for
        seasonal dummies. If not provided, freq is read from index if
        available.
    constant : bool, default False
        Whether to include a constant.
    order : int, default 0
        The order of the tim trend to include. For example, 2 will include
        both linear and quadratic terms. 0 exclude time trend terms.
    seasonal : bool = False
        Whether to include seasonal dummies
    fourier : int = 0
        The order of the fourier terms to included.
    additional_terms : Sequence[DeterministicTerm]
        A sequence of additional deterministic terms to include in the process.
    drop : bool, default False
        A flag indicating to check for perfect collinearity and to drop any
        linearly dependent terms.

    See Also
    --------
    TimeTrend
    Seasonality
    Fourier
    CalendarTimeTrend
    CalendarSeasonality
    CalendarFourier

    Notes
    -----
    See the notebook `Deterministic Terms in Time Series Models
    <../examples/notebooks/generated/deterministics.html>`__ for an overview.

    Examples
    --------
    >>> from statsmodels.tsa.deterministic import DeterministicProcess
    >>> from pandas import date_range
    >>> index = date_range("2000-1-1", freq="M", periods=240)

    First a determinstic process with a constant and quadratic time trend.

    >>> dp = DeterministicProcess(index, constant=True, order=2)
    >>> dp.in_sample().head(3)
                const  trend  trend_squared
    2000-01-31    1.0    1.0            1.0
    2000-02-29    1.0    2.0            4.0
    2000-03-31    1.0    3.0            9.0

    Seasonal dummies are included by setting seasonal to True.

    >>> dp = DeterministicProcess(index, constant=True, seasonal=True)
    >>> dp.in_sample().iloc[:3,:5]
                const  s(2,12)  s(3,12)  s(4,12)  s(5,12)
    2000-01-31    1.0      0.0      0.0      0.0      0.0
    2000-02-29    1.0      1.0      0.0      0.0      0.0
    2000-03-31    1.0      0.0      1.0      0.0      0.0

    Fourier components can be used to alternatively capture seasonal patterns,

    >>> dp = DeterministicProcess(index, constant=True, fourier=2)
    >>> dp.in_sample().head(3)
                const  sin(1,12)  cos(1,12)  sin(2,12)  cos(2,12)
    2000-01-31    1.0   0.000000   1.000000   0.000000        1.0
    2000-02-29    1.0   0.500000   0.866025   0.866025        0.5
    2000-03-31    1.0   0.866025   0.500000   0.866025       -0.5

    Multiple Seasonalities can be captured using additional terms.

    >>> from statsmodels.tsa.deterministic import Fourier
    >>> index = date_range("2000-1-1", freq="D", periods=5000)
    >>> fourier = Fourier(period=365.25, order=1)
    >>> dp = DeterministicProcess(index, period=3, constant=True,
    ...                           seasonal=True, additional_terms=[fourier])
    >>> dp.in_sample().head(3)
                const  s(2,3)  s(3,3)  sin(1,365.25)  cos(1,365.25)
    2000-01-01    1.0     0.0     0.0       0.000000       1.000000
    2000-01-02    1.0     1.0     0.0       0.017202       0.999852
    2000-01-03    1.0     0.0     1.0       0.034398       0.999408
    """

    def __init__(self, index: Union[Sequence[Hashable], pd.Index], *, period: Optional[Union[float, int]]=None, constant: bool=False, order: int=0, seasonal: bool=False, fourier: int=0, additional_terms: Sequence[DeterministicTerm]=(), drop: bool=False):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        self._index = index
        self._deterministic_terms: List[DeterministicTerm] = []
        self._extendable = False
        self._index_freq = None
        self._validate_index()
        period = float_like(period, 'period', optional=True)
        self._constant = constant = bool_like(constant, 'constant')
        self._order = required_int_like(order, 'order')
        self._seasonal = seasonal = bool_like(seasonal, 'seasonal')
        self._fourier = required_int_like(fourier, 'fourier')
        additional_terms = tuple(additional_terms)
        self._cached_in_sample = None
        self._drop = bool_like(drop, 'drop')
        self._additional_terms = additional_terms
        if constant or order:
            self._deterministic_terms.append(TimeTrend(constant, order))
        if seasonal and fourier:
            raise ValueError('seasonal and fourier can be initialized through the constructor since these will be necessarily perfectly collinear. Instead, you can pass additional components using the additional_terms input.')
        if (seasonal or fourier) and period is None:
            if period is None:
                self._period = period = freq_to_period(self._index_freq)
        if seasonal:
            period = required_int_like(period, 'period')
            self._deterministic_terms.append(Seasonality(period))
        elif fourier:
            period = float_like(period, 'period')
            assert period is not None
            self._deterministic_terms.append(Fourier(period, order=fourier))
        for term in additional_terms:
            if not isinstance(term, DeterministicTerm):
                raise TypeError('All additional terms must be instances of subsclasses of DeterministicTerm')
            if term not in self._deterministic_terms:
                self._deterministic_terms.append(term)
            else:
                raise ValueError('One or more terms in additional_terms has been added through the parameters of the constructor. Terms must be unique.')
        self._period = period
        self._retain_cols: Optional[List[Hashable]] = None

    @property
    def index(self) -> pd.Index:
        """The index of the process"""
        pass

    @property
    def terms(self) -> List[DeterministicTerm]:
        """The deterministic terms included in the process"""
        pass

    def range(self, start: Union[IntLike, DateLike, str], stop: Union[IntLike, DateLike, str]) -> pd.DataFrame:
        """
        Deterministic terms spanning a range of observations

        Parameters
        ----------
        start : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The first observation.
        stop : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The final observation. Inclusive to match most prediction
            function in statsmodels.

        Returns
        -------
        DataFrame
            A data frame of deterministic terms
        """
        pass

    def apply(self, index):
        """
        Create an identical determinstic process with a different index

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DeterministicProcess
            The deterministic process applied to a different index
        """
        pass