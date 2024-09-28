from optparse import OptionParser
import statsmodels.api as sm
import scipy as sp
from scipy import linalg
from scipy import stats
docstr = '\nDemonstrates l1 regularization for likelihood models.\nUse different models by setting mode = mnlogit, logit, or probit.\n\nExamples\n-------\n$ python demo.py --get_l1_slsqp_results  logit\n\n>>> import demo\n>>> demo.run_demo(\'logit\')\n\nThe Story\n---------\nThe maximum likelihood (ML) solution works well when the number of data\npoints is large and the noise is small.  When the ML solution starts\n"breaking", the regularized solution should do better.\n\nThe l1 Solvers\n--------------\nThe solvers are slower than standard Newton, and sometimes have\n    convergence issues Nonetheless, the final solution makes sense and\n    is often better than the ML solution.\nThe standard l1 solver is fmin_slsqp and is included with scipy.  It\n    sometimes has trouble verifying convergence when the data size is\n    large.\nThe l1_cvxopt_cp solver is part of CVXOPT and this package needs to be\n    installed separately.  It works well even for larger data sizes.\n'

def main():
    """
    Provides a CLI for the demo.
    """
    pass

def run_demo(mode, base_alpha=0.01, N=500, get_l1_slsqp_results=False, get_l1_cvxopt_results=False, num_nonconst_covariates=10, noise_level=0.2, cor_length=2, num_zero_params=8, num_targets=3, print_summaries=False, save_arrays=False, load_old_arrays=False):
    """
    Run the demo and print results.

    Parameters
    ----------
    mode : str
        either 'logit', 'mnlogit', or 'probit'
    base_alpha :  Float
        Size of regularization param (the param actually used will
        automatically scale with data size in this demo)
    N : int
        Number of data points to generate for fit
    get_l1_slsqp_results : bool,
        Do an l1 fit using slsqp.
    get_l1_cvxopt_results : bool
        Do an l1 fit using cvxopt
    num_nonconst_covariates : int
        Number of covariates that are not constant
        (a constant will be prepended)
    noise_level : float (non-negative)
        Level of the noise relative to signal
    cor_length : float (non-negative)
        Correlation length of the (Gaussian) independent variables
    num_zero_params : int
        Number of parameters equal to zero for every target in logistic
        regression examples.
    num_targets : int
        Number of choices for the endogenous response in multinomial logit
        example
    print_summaries : bool
        print the full fit summary.
    save_arrays : bool
        Save exog/endog/true_params to disk for future use.
    load_old_arrays
        Load exog/endog/true_params arrays from disk.
    """
    pass

def run_solvers(model, true_params, alpha, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries):
    """
    Runs the solvers using the specified settings and returns a result string.
    Works the same for any l1 penalized likelihood model.
    """
    pass

def get_summary_str(results, true_params, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries):
    """
    Gets a string summarizing the results.
    """
    pass

def get_RMSE(results, true_params):
    """
    Gets the (normalized) root mean square error.
    """
    pass

def get_logit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    pass

def get_probit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    pass

def get_exog(N, num_nonconst_covariates, cor_length):
    """
    Returns an exog array with correlations determined by cor_length.
    The covariance matrix of exog will have (asymptotically, as
    :math:'N\\to\\inf')
    .. math:: Cov[i,j] = \\exp(-|i-j| / cor_length)

    Higher cor_length makes the problem more ill-posed, and easier to screw
        up with noise.
    BEWARE:  With very long correlation lengths, you often get a singular KKT
        matrix (during the l1_cvxopt_cp fit)
    """
    pass
if __name__ == '__main__':
    main()