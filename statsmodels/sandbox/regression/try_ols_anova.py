""" convenience functions for ANOVA type analysis with OLS

Note: statistical results of ANOVA are not checked, OLS is
checked but not whether the reported results are the ones used
in ANOVA

includes form2design for creating dummy variables

TODO:
 * ...
 *

"""
from statsmodels.compat.python import lmap
import numpy as np
import statsmodels.api as sm

def data2dummy(x, returnall=False):
    """convert array of categories to dummy variables
    by default drops dummy variable for last category
    uses ravel, 1d only"""
    pass

def data2proddummy(x):
    """creates product dummy variables from 2 columns of 2d array

    drops last dummy variable, but not from each category
    singular with simple dummy variable but not with constant

    quickly written, no safeguards

    """
    pass

def data2groupcont(x1, x2):
    """create dummy continuous variable

    Parameters
    ----------
    x1 : 1d array
        label or group array
    x2 : 1d array (float)
        continuous variable

    Notes
    -----
    useful for group specific slope coefficients in regression
    """
    pass
anova_str0 = '\nANOVA statistics (model sum of squares excludes constant)\nSource    DF  Sum Squares   Mean Square    F Value    Pr > F\nModel     %(df_model)i        %(ess)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f\nError     %(df_resid)i     %(ssr)f       %(mse_resid)f\nCTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f\n\nR squared  %(rsquared)f\n'
anova_str = '\nANOVA statistics (model sum of squares includes constant)\nSource    DF  Sum Squares   Mean Square    F Value    Pr > F\nModel     %(df_model)i      %(ssmwithmean)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f\nError     %(df_resid)i     %(ssr)f       %(mse_resid)f\nCTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f\n\nR squared  %(rsquared)f\n'

def anovadict(res):
    """update regression results dictionary with ANOVA specific statistics

    not checked for completeness
    """
    pass

def form2design(ss, data):
    """convert string formula to data dictionary

    ss : str
     * I : add constant
     * varname : for simple varnames data is used as is
     * F:varname : create dummy variables for factor varname
     * P:varname1*varname2 : create product dummy variables for
       varnames
     * G:varname1*varname2 : create product between factor and
       continuous variable
    data : dict or structured array
       data set, access of variables by name as in dictionaries

    Returns
    -------
    vars : dictionary
        dictionary of variables with converted dummy variables
    names : list
        list of names, product (P:) and grouped continuous
        variables (G:) have name by joining individual names
        sorted according to input

    Examples
    --------
    >>> xx, n = form2design('I a F:b P:c*d G:c*f', testdata)
    >>> xx.keys()
    ['a', 'b', 'const', 'cf', 'cd']
    >>> n
    ['const', 'a', 'b', 'cd', 'cf']

    Notes
    -----

    with sorted dict, separate name list would not be necessary
    """
    pass

def dropname(ss, li):
    """drop names from a list of strings,
    names to drop are in space delimited list
    does not change original list
    """
    pass
if __name__ == '__main__':
    nobs = 1000
    testdataint = np.random.randint(3, size=(nobs, 4)).view([('a', int), ('b', int), ('c', int), ('d', int)])
    testdatacont = np.random.normal(size=(nobs, 2)).view([('e', float), ('f', float)])
    import numpy.lib.recfunctions
    dt2 = numpy.lib.recfunctions.zip_descr((testdataint, testdatacont), flatten=True)
    testdata = np.empty((nobs, 1), dt2)
    for name in testdataint.dtype.names:
        testdata[name] = testdataint[name]
    for name in testdatacont.dtype.names:
        testdata[name] = testdatacont[name]
    if 0:
        xx, n = form2design('F:a', testdata)
        print(xx)
        print(form2design('P:a*b', testdata))
        print(data2proddummy(np.c_[testdata['a'], testdata['b']]))
        xx, names = form2design('a F:b P:c*d', testdata)
    xx, names = form2design('I a F:b P:c*d', testdata)
    xx, names = form2design('I a F:b P:c*d G:a*e f', testdata)
    X = np.column_stack([xx[nn] for nn in names])
    y = X.sum(1) + 0.01 * np.random.normal(size=nobs)
    rest1 = sm.OLS(y, X).fit()
    print(rest1.params)
    print(anova_str % anovadict(rest1))
    X = np.column_stack([xx[nn] for nn in dropname('ae f', names)])
    y = X.sum(1) + 0.01 * np.random.normal(size=nobs)
    rest1 = sm.OLS(y, X).fit()
    print(rest1.params)
    print(anova_str % anovadict(rest1))
    dt_b = np.dtype([('breed', int), ('sex', int), ('litter', int), ('pen', int), ('pig', int), ('age', float), ('bage', float), ('y', float)])
    dta = np.genfromtxt('dftest3.data', dt_b, missing='.', usemask=True)
    print('missing', [dta.mask[k].sum() for k in dta.dtype.names])
    m = dta.mask.view(bool)
    droprows = m.reshape(-1, len(dta.dtype.names)).any(1)
    dta_use_b1 = dta[~droprows, :].data
    print(dta_use_b1.shape)
    print(dta_use_b1.dtype)
    xx_b1, names_b1 = form2design('I F:sex age', dta_use_b1)
    X_b1 = np.column_stack([xx_b1[nn] for nn in dropname('', names_b1)])
    y_b1 = dta_use_b1['y']
    rest_b1 = sm.OLS(y_b1, X_b1).fit()
    print(rest_b1.params)
    print(anova_str % anovadict(rest_b1))
    allexog = ' '.join(dta.dtype.names[:-1])
    xx_b1a, names_b1a = form2design('I F:breed F:sex F:litter F:pen age bage', dta_use_b1)
    X_b1a = np.column_stack([xx_b1a[nn] for nn in dropname('', names_b1a)])
    y_b1a = dta_use_b1['y']
    rest_b1a = sm.OLS(y_b1a, X_b1a).fit()
    print(rest_b1a.params)
    print(anova_str % anovadict(rest_b1a))
    for dropn in names_b1a:
        print(('\nResults dropping', dropn))
        X_b1a_ = np.column_stack([xx_b1a[nn] for nn in dropname(dropn, names_b1a)])
        y_b1a_ = dta_use_b1['y']
        rest_b1a_ = sm.OLS(y_b1a_, X_b1a_).fit()
        print(anova_str % anovadict(rest_b1a_))