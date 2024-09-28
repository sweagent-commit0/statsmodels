"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state

def _debyem1_expansion(x):
    """Debye function minus 1, Taylor series approximation around zero

    function is not used
    """
    pass

def tau_frank(theta):
    """Kendall's tau for Frank Copula

    This uses Taylor series expansion for theta <= 1.

    Parameters
    ----------
    theta : float
        Parameter of the Frank copula. (not vectorized)

    Returns
    -------
    tau : float, tau for given theta
    """
    pass

class ArchimedeanCopula(Copula):
    """Base class for Archimedean copulas

    Parameters
    ----------
    transform : instance of transformation class
        Archimedean generator with required methods including first and second
        derivatives
    args : tuple
        Optional copula parameters. Copula parameters can be either provided
        when creating the instance or as arguments when calling methods.
    k_dim : int
        Dimension, number of components in the multivariate random variable.
        Currently only bivariate copulas are verified. Support for more than
        2 dimension is incomplete.
    """

    def __init__(self, transform, args=(), k_dim=2):
        super().__init__(k_dim=k_dim)
        self.args = args
        self.transform = transform
        self.k_args = 1

    def cdf(self, u, args=()):
        """Evaluate cdf of Archimedean copula."""
        pass

    def pdf(self, u, args=()):
        """Evaluate pdf of Archimedean copula."""
        pass

    def logpdf(self, u, args=()):
        """Evaluate log pdf of multivariate Archimedean copula."""
        pass

class ClaytonCopula(ArchimedeanCopula):
    """Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    .. math::

        C_\\theta(u,v) = \\left[ \\max\\left\\{ u^{-\\theta} + v^{-\\theta} -1 ;
        0 \\right\\} \\right]^{-1/\\theta}

    with :math:`\\theta\\in[-1,\\infty)\\backslash\\{0\\}`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfClayton(), args=args, k_dim=k_dim)
        if theta is not None:
            if theta <= -1 or theta == 0:
                raise ValueError('Theta must be > -1 and !=0')
        self.theta = theta

class FrankCopula(ArchimedeanCopula):
    """Frank copula.

    Dependence is symmetric.

    .. math::

        C_\\theta(\\mathbf{u}) = -\\frac{1}{\\theta} \\log \\left[ 1-
        \\frac{ \\prod_j (1-\\exp(- \\theta u_j)) }{ (1 - \\exp(-\\theta)-1)^{d -
        1} } \\right]

    with :math:`\\theta\\in \\mathbb{R}\\backslash\\{0\\}, \\mathbf{u} \\in [0, 1]^d`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfFrank(), args=args, k_dim=k_dim)
        if theta is not None:
            if theta == 0:
                raise ValueError('Theta must be !=0')
        self.theta = theta

    def cdfcond_2g1(self, u, args=()):
        """Conditional cdf of second component given the value of first.
        """
        pass

    def ppfcond_2g1(self, q, u1, args=()):
        """Conditional pdf of second component given the value of first.
        """
        pass

class GumbelCopula(ArchimedeanCopula):
    """Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\\theta(u,v) = \\exp\\!\\left[ -\\left( (-\\log(u))^\\theta +
        (-\\log(v))^\\theta \\right)^{1/\\theta} \\right]

    with :math:`\\theta\\in[1,\\infty)`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfGumbel(), args=args, k_dim=k_dim)
        if theta is not None:
            if theta <= 1:
                raise ValueError('Theta must be > 1')
        self.theta = theta