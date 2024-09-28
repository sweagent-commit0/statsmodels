""" Pickand's dependence functions as generators for EV-copulas


Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess

class PickandDependence:

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def deriv(self, t, *args):
        """First derivative of the dependence function

        implemented through numerical differentiation
        """
        pass

    def deriv2(self, t, *args):
        """Second derivative of the dependence function

        implemented through numerical differentiation
        """
        pass

class AsymLogistic(PickandDependence):
    """asymmetric logistic model of Tawn 1988

    special case: a1=a2=1 : Gumbel

    restrictions:
     - theta in (0,1]
     - a1, a2 in [0,1]
    """
    k_args = 3
transform_tawn = AsymLogistic()

class AsymNegLogistic(PickandDependence):
    """asymmetric negative logistic model of Joe 1990

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - theta in (0,inf)
     - a1, a2 in (0,1]
    """
    k_args = 3
transform_joe = AsymNegLogistic()

class AsymMixed(PickandDependence):
    """asymmetric mixed model of Tawn 1988

    special case:  k=0, theta in [0,1] : symmetric mixed model of
        Tiago de Oliveira 1980

    restrictions:
     - theta > 0
     - theta + 3*k > 0
     - theta + k <= 1
     - theta + 2*k <= 1
    """
    k_args = 2
transform_tawn2 = AsymMixed()

class AsymBiLogistic(PickandDependence):
    """bilogistic model of Coles and Tawn 1994, Joe, Smith and Weissman 1992

    restrictions:
     - (beta, delta) in (0,1)^2 or
     - (beta, delta) in (-inf,0)^2

    not vectorized because of numerical integration
    """
    k_args = 2
transform_bilogistic = AsymBiLogistic()

class HR(PickandDependence):
    """model of Huesler Reiss 1989

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - lambda in (0,inf)
    """
    k_args = 1
transform_hr = HR()

class TEV(PickandDependence):
    """t-EV model of Demarta and McNeil 2005

    restrictions:
     - rho in (-1,1)
     - x > 0
    """
    k_args = 2
transform_tev = TEV()