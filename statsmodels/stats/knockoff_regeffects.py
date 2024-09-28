import numpy as np

class RegressionEffects:
    """
    Base class for regression effects used in RegressionFDR.

    Any implementation of the class must provide a method called
    'stats' that takes a RegressionFDR object and returns effect sizes
    for the model coefficients.  Greater values for these statistics
    imply greater evidence that the effect is real.

    Knockoff effect sizes are based on fitting the regression model to
    an extended design matrix [X X'], where X' is a design matrix with
    the same shape as the actual design matrix X.  The construction of
    X' guarantees that there are no true associations between the
    columns of X' and the dependent variable of the regression.  If X
    has p columns, then the effect size of covariate j is based on the
    strength of the estimated association for coefficient j compared
    to the strength of the estimated association for coefficient p+j.
    """

class CorrelationEffects(RegressionEffects):
    """
    Marginal correlation effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the marginal correlation approach to
    constructing test statistics for a knockoff analysis, as
    described under (1) in section 2.2 of the Barber and Candes
    paper.
    """

class ForwardEffects(RegressionEffects):
    """
    Forward selection effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    pursuit : bool
        If True, 'basis pursuit' is used, which amounts to performing
        a full regression at each selection step to adjust the working
        residual vector.  If False (the default), the residual is
        adjusted by regressing out each selected variable marginally.
        Setting pursuit=True will be considerably slower, but may give
        better results when exog is not orthogonal.

    Notes
    -----
    This class implements the forward selection approach to
    constructing test statistics for a knockoff analysis, as
    described under (5) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, pursuit):
        self.pursuit = pursuit

class OLSEffects(RegressionEffects):
    """
    OLS regression for knockoff analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the ordinary least squares regression
    approach to constructing test statistics for a knockoff analysis,
    as described under (2) in section 2.2 of the Barber and Candes
    paper.
    """

class RegModelEffects(RegressionEffects):
    """
    Use any regression model for Regression FDR analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    model_cls : class
        Any model with appropriate fit or fit_regularized
        functions
    regularized : bool
        If True, use fit_regularized to fit the model
    model_kws : dict
        Keywords passed to model initializer
    fit_kws : dict
        Dictionary of keyword arguments for fit or fit_regularized
    """

    def __init__(self, model_cls, regularized=False, model_kws=None, fit_kws=None):
        self.model_cls = model_cls
        self.regularized = regularized
        self.model_kws = model_kws if model_kws is not None else {}
        self.fit_kws = fit_kws if fit_kws is not None else {}