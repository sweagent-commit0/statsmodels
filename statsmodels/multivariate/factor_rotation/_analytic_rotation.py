"""
This file contains analytic implementations of rotation methods.
"""
import numpy as np
import scipy as sp

def target_rotation(A, H, full_rank=False):
    """
    Analytically performs orthogonal rotations towards a target matrix,
    i.e., we minimize:

    .. math::
        \\phi(L) =\\frac{1}{2}\\|AT-H\\|^2.

    where :math:`T` is an orthogonal matrix. This problem is also known as
    an orthogonal Procrustes problem.

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:

    .. math::
        T = (A^*HH^*A)^{-\\frac{1}{2}}A^*H,

    see Green (1952). In other cases the solution is given by :math:`T = UV`,
    where :math:`U` and :math:`V` result from the singular value decomposition
    of :math:`A^*H`:

    .. math::
        A^*H = U\\Sigma V,

    see Schonemann (1966).

    Parameters
    ----------
    A : numpy matrix (default None)
        non rotated factors
    H : numpy matrix
        target matrix
    full_rank : bool (default FAlse)
        if set to true full rank is assumed

    Returns
    -------
    The matrix :math:`T`.

    References
    ----------
    [1] Green (1952, Psychometrika) - The orthogonal approximation of an
    oblique structure in factor analysis

    [2] Schonemann (1966) - A generalized solution of the orthogonal
    procrustes problem

    [3] Gower, Dijksterhuis (2004) - Procrustes problems
    """
    pass

def procrustes(A, H):
    """
    Analytically solves the following Procrustes problem:

    .. math::
        \\phi(L) =\\frac{1}{2}\\|AT-H\\|^2.

    (With no further conditions on :math:`H`)

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:

    .. math::
        T = (A^*HH^*A)^{-\\frac{1}{2}}A^*H,

    see Navarra, Simoncini (2010).

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    H : numpy matrix
        target matrix
    full_rank : bool (default False)
        if set to true full rank is assumed

    Returns
    -------
    The matrix :math:`T`.

    References
    ----------
    [1] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions
    for climate data analysis
    """
    pass

def promax(A, k=2):
    """
    Performs promax rotation of the matrix :math:`A`.

    This method was not very clear to me from the literature, this
    implementation is as I understand it should work.

    Promax rotation is performed in the following steps:

    * Determine varimax rotated patterns :math:`V`.

    * Construct a rotation target matrix :math:`|V_{ij}|^k/V_{ij}`

    * Perform procrustes rotation towards the target to obtain T

    * Determine the patterns

    First, varimax rotation a target matrix :math:`H` is determined with
    orthogonal varimax rotation.
    Then, oblique target rotation is performed towards the target.

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    k : float
        parameter, should be positive

    References
    ----------
    [1] Browne (2001) - An overview of analytic rotation in exploratory
    factor analysis

    [2] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions
    for climate data analysis
    """
    pass