import numpy as np
LOWER_BOUND = np.sqrt(np.finfo(float).eps)

class HoltWintersArgs:

    def __init__(self, xi, p, bounds, y, m, n, transform=False):
        self._xi = xi
        self._p = p
        self._bounds = bounds
        self._y = y
        self._lvl = np.empty(n)
        self._b = np.empty(n)
        self._s = np.empty(n + m - 1)
        self._m = m
        self._n = n
        self._transform = transform

def to_restricted(p, sel, bounds):
    """
    Transform parameters from the unrestricted [0,1] space
    to satisfy both the bounds and the 2 constraints
    beta <= alpha and gamma <= (1-alpha)

    Parameters
    ----------
    p : ndarray
        The parameters to transform
    sel : ndarray
        Array indicating whether a parameter is being estimated. If not
        estimated, not transformed.
    bounds : ndarray
        2-d array of bounds where bound for element i is in row i
        and stored as [lb, ub]

    Returns
    -------

    """
    pass

def to_unrestricted(p, sel, bounds):
    """
    Transform parameters to the unrestricted [0,1] space

    Parameters
    ----------
    p : ndarray
        Parameters that strictly satisfy the constraints

    Returns
    -------
    ndarray
        Parameters all in (0,1)
    """
    pass

def holt_init(x, hw_args: HoltWintersArgs):
    """
    Initialization for the Holt Models
    """
    pass

def holt__(x, hw_args: HoltWintersArgs):
    """
    Simple Exponential Smoothing
    Minimization Function
    (,)
    """
    pass

def holt_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped
    Minimization Function
    (M,) & (Md,)
    """
    pass

def holt_add_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped
    Minimization Function
    (A,) & (Ad,)
    """
    pass

def holt_win_init(x, hw_args: HoltWintersArgs):
    """Initialization for the Holt Winters Seasonal Models"""
    pass

def holt_win__mul(x, hw_args: HoltWintersArgs):
    """
    Multiplicative Seasonal
    Minimization Function
    (,M)
    """
    pass

def holt_win__add(x, hw_args: HoltWintersArgs):
    """
    Additive Seasonal
    Minimization Function
    (,A)
    """
    pass

def holt_win_add_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped with Multiplicative Seasonal
    Minimization Function
    (A,M) & (Ad,M)
    """
    pass

def holt_win_mul_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal
    Minimization Function
    (M,M) & (Md,M)
    """
    pass

def holt_win_add_add_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped with Additive Seasonal
    Minimization Function
    (A,A) & (Ad,A)
    """
    pass

def holt_win_mul_add_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped with Additive Seasonal
    Minimization Function
    (M,A) & (M,Ad)
    """
    pass