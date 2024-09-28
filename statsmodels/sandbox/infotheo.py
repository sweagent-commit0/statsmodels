"""
Information Theoretic and Entropy Measures

References
----------
Golan, As. 2008. "Information and Entropy Econometrics -- A Review and
    Synthesis." Foundations And Trends in Econometrics 2(1-2), 1-145.

Golan, A., Judge, G., and Miller, D.  1996.  Maximum Entropy Econometrics.
    Wiley & Sons, Chichester.
"""
from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp

def logsumexp(a, axis=None):
    """
    Compute the log of the sum of exponentials log(e^{a_1}+...e^{a_n}) of a

    Avoids numerical overflow.

    Parameters
    ----------
    a : array_like
        The vector to exponentiate and sum
    axis : int, optional
        The axis along which to apply the operation.  Defaults is None.

    Returns
    -------
    sum(log(exp(a)))

    Notes
    -----
    This function was taken from the mailing list
    http://mail.scipy.org/pipermail/scipy-user/2009-October/022931.html

    This should be superceded by the ufunc when it is finished.
    """
    pass

def _isproperdist(X):
    """
    Checks to see if `X` is a proper probability distribution
    """
    pass

def discretize(X, method='ef', nbins=None):
    """
    Discretize `X`

    Parameters
    ----------
    bins : int, optional
        Number of bins.  Default is floor(sqrt(N))
    method : str
        "ef" is equal-frequency binning
        "ew" is equal-width binning

    Examples
    --------
    """
    pass

def logbasechange(a, b):
    """
    There is a one-to-one transformation of the entropy value from
    a log base b to a log base a :

    H_{b}(X)=log_{b}(a)[H_{a}(X)]

    Returns
    -------
    log_{b}(a)
    """
    pass

def natstobits(X):
    """
    Converts from nats to bits
    """
    pass

def bitstonats(X):
    """
    Converts from bits to nats
    """
    pass

def shannonentropy(px, logbase=2):
    """
    This is Shannon's entropy

    Parameters
    ----------
    logbase, int or np.e
        The base of the log
    px : 1d or 2d array_like
        Can be a discrete probability distribution, a 2d joint distribution,
        or a sequence of probabilities.

    Returns
    -----
    For log base 2 (bits) given a discrete distribution
        H(p) = sum(px * log2(1/px) = -sum(pk*log2(px)) = E[log2(1/p(X))]

    For log base 2 (bits) given a joint distribution
        H(px,py) = -sum_{k,j}*w_{kj}log2(w_{kj})

    Notes
    -----
    shannonentropy(0) is defined as 0
    """
    pass

def shannoninfo(px, logbase=2):
    """
    Shannon's information

    Parameters
    ----------
    px : float or array_like
        `px` is a discrete probability distribution

    Returns
    -------
    For logbase = 2
    np.log2(px)
    """
    pass

def condentropy(px, py, pxpy=None, logbase=2):
    """
    Return the conditional entropy of X given Y.

    Parameters
    ----------
    px : array_like
    py : array_like
    pxpy : array_like, optional
        If pxpy is None, the distributions are assumed to be independent
        and conendtropy(px,py) = shannonentropy(px)
    logbase : int or np.e

    Returns
    -------
    sum_{kj}log(q_{j}/w_{kj}

    where q_{j} = Y[j]
    and w_kj = X[k,j]
    """
    pass

def mutualinfo(px, py, pxpy, logbase=2):
    """
    Returns the mutual information between X and Y.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like
        The joint probability distribution of random variables X and Y.
        Note that if X and Y are independent then the mutual information
        is zero.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    shannonentropy(px) - condentropy(px,py,pxpy)
    """
    pass

def corrent(px, py, pxpy, logbase=2):
    """
    An information theoretic correlation measure.

    Reflects linear and nonlinear correlation between two random variables
    X and Y, characterized by the discrete probability distributions px and py
    respectively.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,logbase=logbase)

    Notes
    -----
    This is also equivalent to

    corrent(px,py,pxpy) = 1 - condent(px,py,pxpy)/shannonentropy(py)
    """
    pass

def covent(px, py, pxpy, logbase=2):
    """
    An information theoretic covariance measure.

    Reflects linear and nonlinear correlation between two random variables
    X and Y, characterized by the discrete probability distributions px and py
    respectively.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    condent(px,py,pxpy,logbase=logbase) + condent(py,px,pxpy,
            logbase=logbase)

    Notes
    -----
    This is also equivalent to

    covent(px,py,pxpy) = condent(px,py,pxpy) + condent(py,px,pxpy)
    """
    pass

def renyientropy(px, alpha=1, logbase=2, measure='R'):
    """
    Renyi's generalized entropy

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X.  Note that
        px is assumed to be a proper probability distribution.
    logbase : int or np.e, optional
        Default is 2 (bits)
    alpha : float or inf
        The order of the entropy.  The default is 1, which in the limit
        is just Shannon's entropy.  2 is Renyi (Collision) entropy.  If
        the string "inf" or numpy.inf is specified the min-entropy is returned.
    measure : str, optional
        The type of entropy measure desired.  'R' returns Renyi entropy
        measure.  'T' returns the Tsallis entropy measure.

    Returns
    -------
    1/(1-alpha)*log(sum(px**alpha))

    In the limit as alpha -> 1, Shannon's entropy is returned.

    In the limit as alpha -> inf, min-entropy is returned.
    """
    pass

def gencrossentropy(px, py, pxpy, alpha=1, logbase=2, measure='T'):
    """
    Generalized cross-entropy measures.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)
    measure : str, optional
        The measure is the type of generalized cross-entropy desired. 'T' is
        the cross-entropy version of the Tsallis measure.  'CR' is Cressie-Read
        measure.
    """
    pass
if __name__ == '__main__':
    print('From Golan (2008) "Information and Entropy Econometrics -- A Review and Synthesis')
    print('Table 3.1')
    X = [0.2, 0.2, 0.2, 0.2, 0.2]
    Y = [0.322, 0.072, 0.511, 0.091, 0.004]
    for i in X:
        print(shannoninfo(i))
    for i in Y:
        print(shannoninfo(i))
    print(shannonentropy(X))
    print(shannonentropy(Y))
    p = [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    plt.subplot(111)
    plt.ylabel('Information')
    plt.xlabel('Probability')
    x = np.linspace(0, 1, 100001)
    plt.plot(x, shannoninfo(x))
    plt.subplot(111)
    plt.ylabel('Entropy')
    plt.xlabel('Probability')
    x = np.linspace(0, 1, 101)
    plt.plot(x, lmap(shannonentropy, lzip(x, 1 - x)))
    w = np.array([[0, 0, 1.0 / 3], [1 / 9.0, 1 / 9.0, 1 / 9.0], [1 / 18.0, 1 / 9.0, 1 / 6.0]])
    px = w.sum(0)
    py = w.sum(1)
    H_X = shannonentropy(px)
    H_Y = shannonentropy(py)
    H_XY = shannonentropy(w)
    H_XgivenY = condentropy(px, py, w)
    H_YgivenX = condentropy(py, px, w)
    D_YX = logbasechange(2, np.e) * stats.entropy(px, py)
    D_XY = logbasechange(2, np.e) * stats.entropy(py, px)
    I_XY = mutualinfo(px, py, w)
    print('Table 3.3')
    print(H_X, H_Y, H_XY, H_XgivenY, H_YgivenX, D_YX, D_XY, I_XY)
    print('discretize functions')
    X = np.array([21.2, 44.5, 31.0, 19.5, 40.6, 38.7, 11.1, 15.8, 31.9, 25.8, 20.2, 14.2, 24.0, 21.0, 11.3, 18.0, 16.3, 22.2, 7.8, 27.8, 16.3, 35.1, 14.9, 17.1, 28.2, 16.4, 16.5, 46.0, 9.5, 18.8, 32.1, 26.1, 16.1, 7.3, 21.4, 20.0, 29.3, 14.9, 8.3, 22.5, 12.8, 26.9, 25.5, 22.9, 11.2, 20.7, 26.2, 9.3, 10.8, 15.6])
    discX = discretize(X)
    print
    print('Example in section 3.6 of Golan, using table 3.3')
    print("Bounding errors using Fano's inequality")
    print('H(P_{e}) + P_{e}log(K-1) >= H(X|Y)')
    print('or, a weaker inequality')
    print('P_{e} >= [H(X|Y) - 1]/log(K)')
    print('P(x) = %s' % px)
    print('X = 3 has the highest probability, so this is the estimate Xhat')
    pe = 1 - px[2]
    print('The probability of error Pe is 1 - p(X=3) = %0.4g' % pe)
    H_pe = shannonentropy([pe, 1 - pe])
    print('H(Pe) = %0.4g and K=3' % H_pe)
    print('H(Pe) + Pe*log(K-1) = %0.4g >= H(X|Y) = %0.4g' % (H_pe + pe * np.log2(2), H_XgivenY))
    print('or using the weaker inequality')
    print('Pe = %0.4g >= [H(X) - 1]/log(K) = %0.4g' % (pe, (H_X - 1) / np.log2(3)))
    print('Consider now, table 3.5, where there is additional information')
    print('The conditional probabilities of P(X|Y=y) are ')
    w2 = np.array([[0.0, 0.0, 1.0], [1 / 3.0, 1 / 3.0, 1 / 3.0], [1 / 6.0, 1 / 3.0, 1 / 2.0]])
    print(w2)
    print('The probability of error given this information is')
    print('Pe = [H(X|Y) -1]/log(K) = %0.4g' % ((np.mean([0, shannonentropy(w2[1]), shannonentropy(w2[2])]) - 1) / np.log2(3)))
    print('such that more information lowers the error')
    markovchain = np.array([[0.553, 0.284, 0.163], [0.465, 0.312, 0.223], [0.42, 0.322, 0.258]])