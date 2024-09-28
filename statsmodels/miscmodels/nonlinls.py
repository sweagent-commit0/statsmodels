"""Non-linear least squares



Author: Josef Perktold based on scipy.optimize.curve_fit

"""
import numpy as np
from scipy import optimize
from statsmodels.base.model import Model

class Results:
    """just a dummy placeholder for now
    most results from RegressionResults can be used here
    """
    pass

class NonlinearLS(Model):
    """Base class for estimation of a non-linear model with least squares

    This class is supposed to be subclassed, and the subclass has to provide a method
    `_predict` that defines the non-linear function `f(params) that is predicting the endogenous
    variable. The model is assumed to be

    :math: y = f(params) + error

    and the estimator minimizes the sum of squares of the estimated error.

    :math: min_parmas \\sum (y - f(params))**2

    f has to return the prediction for each observation. Exogenous or explanatory variables
    should be accessed as attributes of the class instance, and can be given as arguments
    when the instance is created.

    Warning:
    Weights are not correctly handled yet in the results statistics,
    but included when estimating the parameters.

    similar to scipy.optimize.curve_fit
    API difference: params are array_like not split up, need n_params information

    includes now weights similar to curve_fit
    no general sigma yet (OLS and WLS, but no GLS)

    This is currently holding on to intermediate results that are not necessary
    but useful for testing.

    Fit returns and instance of RegressionResult, in contrast to the linear
    model, results in this case are based on a local approximation, essentially
    y = f(X, params) is replaced by y = grad * params where grad is the Gradient
    or Jacobian with the shape (nobs, nparams). See for example Greene

    Examples
    --------

    class Myfunc(NonlinearLS):

        def _predict(self, params):
            x = self.exog
            a, b, c = params
            return a*np.exp(-b*x) + c

    Ff we have data (y, x), we can create an instance and fit it with

    mymod = Myfunc(y, x)
    myres = mymod.fit(nparams=3)

    and use the non-linear regression results, for example

    myres.params
    myres.bse
    myres.tvalues


    """

    def __init__(self, endog=None, exog=None, weights=None, sigma=None, missing='none'):
        self.endog = endog
        self.exog = exog
        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.ndim < 2:
                self.sigma = sigma
                self.weights = 1.0 / sigma
            else:
                raise ValueError('correlated errors are not handled yet')
        else:
            self.weights = None

    def fit_minimal(self, start_value, **kwargs):
        """minimal fitting with no extra calculations"""
        pass

    def fit_random(self, ntries=10, rvs_generator=None, nparams=None):
        """fit with random starting values

        this could be replaced with a global fitter

        """
        pass

    def jac_predict(self, params):
        """jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        """
        pass

class Myfunc(NonlinearLS):
    pass
if __name__ == '__main__':
    x = np.linspace(0, 4, 50)
    params = np.array([2.5, 1.3, 0.5])
    y0 = func(params, x)
    y = y0 + 0.2 * np.random.normal(size=len(x))
    res = optimize.leastsq(error, params, args=(x, y), full_output=True)
    mod = Myfunc(y, x)
    resmy = mod.fit(nparams=3)
    cf_params, cf_pcov = optimize.curve_fit(func0, x, y)
    cf_bse = np.sqrt(np.diag(cf_pcov))
    print(res[0])
    print(cf_params)
    print(resmy.params)
    print(cf_bse)
    print(resmy.bse)