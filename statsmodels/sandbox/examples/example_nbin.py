"""
Author: Vincent Arel-Bundock <varel@umich.edu>
Date: 2012-08-25

This example file implements 5 variations of the negative binomial regression
model for count data: NB-P, NB-1, NB-2, geometric and left-truncated.

The NBin class inherits from the GenericMaximumLikelihood statsmodels class
which provides automatic numerical differentiation for the score and hessian.

NB-1, NB-2 and geometric are implemented as special cases of the NB-P model
described in Greene (2008) Functional forms for the negative binomial model for
count data. Economics Letters, v99n3.

Tests are included to check how NB-1, NB-2 and geometric coefficient estimates
compare to equivalent models in R. Results usually agree up to the 4th digit.

The NB-P and left-truncated model results have not been compared to other
implementations. Note that NB-P appears to only have been implemented in the
LIMDEP software.
"""
from urllib.request import urlopen
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import digamma
from scipy.stats import nbinom
import pandas
import patsy
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModelResults

def _ll_nbp(y, X, beta, alph, Q):
    """
    Negative Binomial Log-likelihood -- type P

    References:

    Greene, W. 2008. "Functional forms for the negative binomial model
        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.
    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University Press.

    Following notation in Greene (2008), with negative binomial heterogeneity
    parameter :math:`\\alpha`:

    .. math::

        \\lambda_i = exp(X\\beta)\\\\
        \\theta = 1 / \\alpha \\\\
        g_i = \\theta \\lambda_i^Q \\\\
        w_i = g_i/(g_i + \\lambda_i) \\\\
        r_i = \\theta / (\\theta+\\lambda_i) \\\\
        ln \\mathcal{L}_i = ln \\Gamma(y_i+g_i) - ln \\Gamma(1+y_i) + g_iln (r_i) + y_i ln(1-r_i)
    """
    pass

def _ll_nb1(y, X, beta, alph):
    """Negative Binomial regression (type 1 likelihood)"""
    pass

def _ll_nb2(y, X, beta, alph):
    """Negative Binomial regression (type 2 likelihood)"""
    pass

def _ll_geom(y, X, beta):
    """Geometric regression"""
    pass

def _ll_nbt(y, X, beta, alph, C=0):
    """
    Negative Binomial (truncated)

    Truncated densities for count models (Cameron & Trivedi, 2005, 680):

    .. math::

        f(y|\\beta, y \\geq C+1) = \\frac{f(y|\\beta)}{1-F(C|\\beta)}
    """
    pass

class NBin(GenericLikelihoodModel):
    """
    Negative Binomial regression

    Parameters
    ----------
    endog : array_like
        1-d array of the response variable.
    exog : array_like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is
        included in the data.
    ll_type: str
        log-likelihood type
        `nb2`: Negative Binomial type-2 (most common)
        `nb1`: Negative Binomial type-1
        `nbp`: Negative Binomial type-P (Greene, 2008)
        `nbt`: Left-truncated Negative Binomial (type-2)
        `geom`: Geometric regression model
    C: int
        Cut-point for `nbt` model
    """

    def __init__(self, endog, exog, ll_type='nb2', C=0, **kwds):
        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.C = C
        super(NBin, self).__init__(endog, exog, **kwds)
        if ll_type not in ['nb2', 'nb1', 'nbp', 'nbt', 'geom']:
            raise NameError('Valid ll_type are: nb2, nb1, nbp,  nbt, geom')
        self.ll_type = ll_type
        if ll_type == 'geom':
            self.start_params_default = np.zeros(self.exog.shape[1])
        elif ll_type == 'nbp':
            start_mod = NBin(endog, exog, 'nb2')
            start_res = start_mod.fit(disp=False)
            self.start_params_default = np.append(start_res.params, 0)
        else:
            self.start_params_default = np.append(np.zeros(self.exog.shape[1]), 0.5)
        self.start_params_default[0] = np.log(self.endog.mean())
        if ll_type == 'nb1':
            self.ll_func = _ll_nb1
        elif ll_type == 'nb2':
            self.ll_func = _ll_nb2
        elif ll_type == 'geom':
            self.ll_func = _ll_geom
        elif ll_type == 'nbp':
            self.ll_func = _ll_nbp
        elif ll_type == 'nbt':
            self.ll_func = _ll_nbt

class CountResults(GenericLikelihoodModelResults):

    def __init__(self, model, mlefit):
        self.model = model
        self.__dict__.update(mlefit.__dict__)

def _score_nbp(y, X, beta, thet, Q):
    """
    Negative Binomial Score -- type P likelihood from Greene (2007)
    .. math::

        \\lambda_i = exp(X\\beta)\\\\
        g_i = \\theta \\lambda_i^Q \\\\
        w_i = g_i/(g_i + \\lambda_i) \\\\
        r_i = \\theta / (\\theta+\\lambda_i) \\\\
        A_i = \\left [ \\Psi(y_i+g_i) - \\Psi(g_i) + ln w_i \\right ] \\\\
        B_i = \\left [ g_i (1-w_i) - y_iw_i \\right ] \\\\
        \\partial ln \\mathcal{L}_i / \\partial
            \\begin{pmatrix} \\lambda_i \\\\ \\theta \\\\ Q \\end{pmatrix}=
            [A_i+B_i]
            \\begin{pmatrix} Q/\\lambda_i \\\\ 1/\\theta \\\\ ln(\\lambda_i) \\end{pmatrix}
            -B_i
            \\begin{pmatrix} 1/\\lambda_i\\\\ 0 \\\\ 0 \\end{pmatrix} \\\\
        \\frac{\\partial \\lambda}{\\partial \\beta} = \\lambda_i \\mathbf{x}_i \\\\
        \\frac{\\partial \\mathcal{L}_i}{\\partial \\beta} =
            \\left (\\frac{\\partial\\mathcal{L}_i}{\\partial \\lambda_i} \\right )
            \\frac{\\partial \\lambda_i}{\\partial \\beta}
    """
    pass
medpar = pandas.read_csv(urlopen('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/csv/COUNT/medpar.csv'))
mdvis = pandas.read_csv(urlopen('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/csv/COUNT/mdvis.csv'))
'\n# R v2.15.1\nlibrary(MASS)\nlibrary(COUNT)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nmod <- glm.nb(f, medpar)\nsummary(mod)\nCall:\nglm.nb(formula = f, data = medpar, init.theta = 2.243376203,\n    link = log)\n\nDeviance Residuals:\n    Min       1Q   Median       3Q      Max\n-2.4671  -0.9090  -0.2693   0.4320   3.8668\n\nCoefficients:\n              Estimate Std. Error z value Pr(>|z|)\n(Intercept)    2.31028    0.06745  34.253  < 2e-16 ***\nfactor(type)2  0.22125    0.05046   4.385 1.16e-05 ***\nfactor(type)3  0.70616    0.07600   9.292  < 2e-16 ***\nhmo           -0.06796    0.05321  -1.277    0.202\nwhite         -0.12907    0.06836  -1.888    0.059 .\n---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n\n(Dispersion parameter for Negative Binomial(2.2434) family taken to be 1)\n\n    Null deviance: 1691.1  on 1494  degrees of freedom\nResidual deviance: 1568.1  on 1490  degrees of freedom\nAIC: 9607\n\nNumber of Fisher Scoring iterations: 1\n\n\n              Theta:  2.2434\n          Std. Err.:  0.0997\n\n 2 x log-likelihood:  -9594.9530\n'
'\n# R v2.15.1\n# COUNT v1.2.3\nlibrary(COUNT)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nml.nb1(f, medpar)\n\n                 Estimate         SE          Z         LCL         UCL\n(Intercept)    2.34918407 0.06023641 38.9994023  2.23112070  2.46724744\nfactor(type)2  0.16175471 0.04585569  3.5274735  0.07187757  0.25163186\nfactor(type)3  0.41879257 0.06553258  6.3906006  0.29034871  0.54723643\nhmo           -0.04533566 0.05004714 -0.9058592 -0.14342805  0.05275673\nwhite         -0.12951295 0.06071130 -2.1332593 -0.24850710 -0.01051880\nalpha          4.57898241 0.22015968 20.7984603  4.14746943  5.01049539\n'
'\nMASS v7.3-20\nR v2.15.1\nlibrary(MASS)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nmod <- glm(f, family=negative.binomial(1), data=medpar)\nsummary(mod)\nCall:\nglm(formula = f, family = negative.binomial(1), data = medpar)\n\nDeviance Residuals:\n    Min       1Q   Median       3Q      Max\n-1.7942  -0.6545  -0.1896   0.3044   2.6844\n\nCoefficients:\n              Estimate Std. Error t value Pr(>|t|)\n(Intercept)    2.30849    0.07071  32.649  < 2e-16 ***\nfactor(type)2  0.22121    0.05283   4.187 2.99e-05 ***\nfactor(type)3  0.70599    0.08092   8.724  < 2e-16 ***\nhmo           -0.06779    0.05521  -1.228   0.2197\nwhite         -0.12709    0.07169  -1.773   0.0765 .\n---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n\n(Dispersion parameter for Negative Binomial(1) family taken to be 0.5409721)\n\n    Null deviance: 872.29  on 1494  degrees of freedom\nResidual deviance: 811.95  on 1490  degrees of freedom\nAIC: 9927.3\n\nNumber of Fisher Scoring iterations: 5\n'
test_nb2()