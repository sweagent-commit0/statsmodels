"""Tools for working with groups

This provides several functions to work with groups and a Group class that
keeps track of the different representations and has methods to work more
easily with groups.


Author: Josef Perktold,
Author: Nathaniel Smith, recipe for sparse_dummies on scipy user mailing list

Created on Tue Nov 29 15:44:53 2011 : sparse_dummies
Created on Wed Nov 30 14:28:24 2011 : combine_indices
changes: add Group class

Notes
~~~~~

This reverses the class I used before, where the class was for the data and
the group was auxiliary. Here, it is only the group, no data is kept.

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
    selection even if the original groups where defined as arange.

Not all methods and options have been tried out yet after refactoring

need more efficient loop if groups are sorted -> see GroupSorted.group_iter
"""
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex

def combine_indices(groups, prefix='', sep='.', return_labels=False):
    """use np.unique to get integer group indices for product, intersection
    """
    pass

def group_sums(x, group, use_bincount=True):
    """simple bincount version, again

    group : ndarray, integer
        assumed to be consecutive integers

    no dtype checking because I want to raise in that case

    uses loop over columns of x

    for comparison, simple python loop
    """
    pass

def group_sums_dummy(x, group_dummy):
    """sum by groups given group dummy variable

    group_dummy can be either ndarray or sparse matrix
    """
    pass

def dummy_sparse(groups):
    """create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups : ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are
        assumed to be defined as consecutive integers, i.e. range(n_groups)
        where n_groups is the number of group levels. A group level with no
        observations for it will still produce a column of zeros.

    Returns
    -------
    indi : ndarray, int8, 2d (nobs, n_groups)
        an indicator array with one row per observation, that has 1 in the
        column of the group level for that observation

    Examples
    --------

    >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi
    <7x3 sparse matrix of type '<type 'numpy.int8'>'
        with 7 stored elements in Compressed Sparse Row format>
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)


    current behavior with missing groups
    >>> g = np.array([0, 0, 2, 0, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)
    """
    pass

class Group:

    def __init__(self, group, name=''):
        self.name = name
        uni, uni_idx, uni_inv = combine_indices(group)
        self.group_int, self.uni_idx, self.uni = (uni, uni_idx, uni_inv)
        self.n_groups = len(self.uni)
        self.separator = '.'
        self.prefix = self.name
        if self.prefix:
            self.prefix = self.prefix + '='

    def dummy(self, drop_idx=None, sparse=False, dtype=int):
        """
        drop_idx is only available if sparse=False

        drop_idx is supposed to index into uni
        """
        pass

class GroupSorted(Group):

    def __init__(self, group, name=''):
        super(self.__class__, self).__init__(group, name=name)
        idx = (np.nonzero(np.diff(group))[0] + 1).tolist()
        self.groupidx = lzip([0] + idx, idx + [len(group)])

    def lag_indices(self, lag):
        """return the index array for lagged values

        Warning: if k is larger then the number of observations for an
        individual, then no values for that individual are returned.

        TODO: for the unbalanced case, I should get the same truncation for
        the array with lag=0. From the return of lag_idx we would not know
        which individual is missing.

        TODO: do I want the full equivalent of lagmat in tsa?
        maxlag or lag or lags.

        not tested yet
        """
        pass

def _is_hierarchical(x):
    """
    Checks if the first item of an array-like object is also array-like
    If so, we have a MultiIndex and returns True. Else returns False.
    """
    pass

class Grouping:

    def __init__(self, index, names=None):
        """
        index : index-like
            Can be pandas MultiIndex or Index or array-like. If array-like
            and is a MultipleIndex (more than one grouping variable),
            groups are expected to be in each row. E.g., [('red', 1),
            ('red', 2), ('green', 1), ('green', 2)]
        names : list or str, optional
            The names to use for the groups. Should be a str if only
            one grouping variable is used.

        Notes
        -----
        If index is already a pandas Index then there is no copy.
        """
        if isinstance(index, (Index, MultiIndex)):
            if names is not None:
                if hasattr(index, 'set_names'):
                    index.set_names(names, inplace=True)
                else:
                    index.names = names
            self.index = index
        else:
            if _is_hierarchical(index):
                self.index = _make_hierarchical_index(index, names)
            else:
                self.index = Index(index, name=names)
            if names is None:
                names = _make_generic_names(self.index)
                if hasattr(self.index, 'set_names'):
                    self.index.set_names(names, inplace=True)
                else:
                    self.index.names = names
        self.nobs = len(self.index)
        self.nlevels = len(self.index.names)
        self.slices = None

    def reindex(self, index=None, names=None):
        """
        Resets the index in-place.
        """
        pass

    def get_slices(self, level=0):
        """
        Sets the slices attribute to be a list of indices of the sorted
        groups for the first index level. I.e., self.slices[0] is the
        index where each observation is in the first (sorted) group.
        """
        pass

    def count_categories(self, level=0):
        """
        Sets the attribute counts to equal the bincount of the (integer-valued)
        labels.
        """
        pass

    def check_index(self, is_sorted=True, unique=True, index=None):
        """Sanity checks"""
        pass

    def sort(self, data, index=None):
        """Applies a (potentially hierarchical) sort operation on a numpy array
        or pandas series/dataframe based on the grouping index or a
        user-supplied index.  Returns an object of the same type as the
        original data as well as the matching (sorted) Pandas index.
        """
        pass

    def transform_dataframe(self, dataframe, function, level=0, **kwargs):
        """Apply function to each column, by group
        Assumes that the dataframe already has a proper index"""
        pass

    def transform_array(self, array, function, level=0, **kwargs):
        """Apply function to each column, by group
        """
        pass

    def transform_slices(self, array, function, level=0, **kwargs):
        """Apply function to each group. Similar to transform_array but does
        not coerce array to a DataFrame and back and only works on a 1D or 2D
        numpy array. function is called function(group, group_idx, **kwargs).
        """
        pass

    def dummy_sparse(self, level=0):
        """create a sparse indicator from a group array with integer labels

        Parameters
        ----------
        groups : ndarray, int, 1d (nobs,)
            An array of group indicators for each observation. Group levels
            are assumed to be defined as consecutive integers, i.e.
            range(n_groups) where n_groups is the number of group levels.
            A group level with no observations for it will still produce a
            column of zeros.

        Returns
        -------
        indi : ndarray, int8, 2d (nobs, n_groups)
            an indicator array with one row per observation, that has 1 in the
            column of the group level for that observation

        Examples
        --------

        >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi
        <7x3 sparse matrix of type '<type 'numpy.int8'>'
            with 7 stored elements in Compressed Sparse Row format>
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)


        current behavior with missing groups
        >>> g = np.array([0, 0, 2, 0, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)
        """
        pass