""" Extreme Value Copulas
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from .copulas import Copula

def copula_bv_ev(u, transform, args=()):
    """generic bivariate extreme value copula
    """
    pass

class ExtremeValueCopula(Copula):
    """Extreme value copula constructed from Pickand's dependence function.

    Currently only bivariate copulas are available.

    Parameters
    ----------
    transform: instance of transformation class
        Pickand's dependence function with required methods including first
        and second derivatives
    args : tuple
        Optional copula parameters. Copula parameters can be either provided
        when creating the instance or as arguments when calling methods.
    k_dim : int
        Currently only bivariate extreme value copulas are supported.

    Notes
    -----
    currently the following dependence function and copulas are available

    - AsymLogistic
    - AsymNegLogistic
    - AsymMixed
    - HR

    TEV and AsymBiLogistic currently do not have required derivatives for pdf.

    See Also
    --------
    dep_func_ev

    """

    def __init__(self, transform, args=(), k_dim=2):
        super().__init__(k_dim=k_dim)
        self.transform = transform
        self.k_args = transform.k_args
        self.args = args
        if k_dim != 2:
            raise ValueError('Only bivariate EV copulas are available.')

    def cdf(self, u, args=()):
        """Evaluate cdf of bivariate extreme value copula.

        Parameters
        ----------
        u : array_like
            Values of random bivariate random variable, each defined on [0, 1],
            for which cdf is computed.
            Can be two dimensional with multivariate components in columns and
            observation in rows.
        args : tuple
            Required parameters for the copula. The meaning and number of
            parameters in the tuple depends on the specific copula.

        Returns
        -------
        CDF values at evaluation points.
        """
        pass

    def pdf(self, u, args=()):
        """Evaluate pdf of bivariate extreme value copula.

        Parameters
        ----------
        u : array_like
            Values of random bivariate random variable, each defined on [0, 1],
            for which cdf is computed.
            Can be two dimensional with multivariate components in columns and
            observation in rows.
        args : tuple
            Required parameters for the copula. The meaning and number of
            parameters in the tuple depends on the specific copula.

        Returns
        -------
        PDF values at evaluation points.
        """
        pass

    def logpdf(self, u, args=()):
        """Evaluate log-pdf of bivariate extreme value copula.

        Parameters
        ----------
        u : array_like
            Values of random bivariate random variable, each defined on [0, 1],
            for which cdf is computed.
            Can be two dimensional with multivariate components in columns and
            observation in rows.
        args : tuple
            Required parameters for the copula. The meaning and number of
            parameters in the tuple depends on the specific copula.

        Returns
        -------
        Log-pdf values at evaluation points.
        """
        pass

    def conditional_2g1(self, u, args=()):
        """conditional distribution

        not yet implemented

        C2|1(u2|u1) := ∂C(u1, u2) / ∂u1 = C(u1, u2) / u1 * (A(t) − t A'(t))

        where t = np.log(v)/np.log(u*v)
        """
        pass