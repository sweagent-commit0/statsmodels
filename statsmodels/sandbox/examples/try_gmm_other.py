import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import tools
from statsmodels.sandbox.regression.gmm import spec_hausman
from statsmodels.sandbox.regression import gmm
if __name__ == '__main__':
    examples = ['ivols', 'distquant'][:]
    if 'ivols' in examples:
        exampledata = ['ols', 'iv', 'ivfake'][1]
        nobs = nsample = 500
        sige = 3
        corrfactor = 0.025
        x = np.linspace(0, 10, nobs)
        X = tools.add_constant(np.column_stack((x, x ** 2)), prepend=False)
        beta = np.array([1, 0.1, 10])
        if exampledata == 'ols':
            endog, exog, _ = sample_ols(X)
            instrument = exog
        elif exampledata == 'iv':
            endog, exog, instrument = sample_iv(X)
        elif exampledata == 'ivfake':
            endog, exog, instrument = sample_ivfake(X)
        mod = gmm.IVGMM(endog, exog, instrument, nmoms=instrument.shape[1])
        res = mod.fit()
        modgmmols = gmm.IVGMM(endog, exog, exog, nmoms=exog.shape[1])
        resgmmols = modgmmols.fit()
        modgmmiv = gmm.IVGMM(endog, exog, instrument, nmoms=instrument.shape[1])
        resgmmiv = modgmmiv.fitgmm(np.ones(exog.shape[1], float), weights=np.linalg.inv(np.dot(instrument.T, instrument)))
        modls = gmm.IV2SLS(endog, exog, instrument)
        resls = modls.fit()
        modols = OLS(endog, exog)
        resols = modols.fit()
        print('\nIV case')
        print('params')
        print('IV2SLS', resls.params)
        print('GMMIV ', resgmmiv)
        print('GMM   ', res.params)
        print('diff  ', res.params - resls.params)
        print('OLS   ', resols.params)
        print('GMMOLS', resgmmols.params)
        print('\nbse')
        print('IV2SLS', resls.bse)
        print('GMM   ', res.bse)
        print('diff  ', res.bse - resls.bse)
        print('%-diff', resls.bse / res.bse * 100 - 100)
        print('OLS   ', resols.bse)
        print('GMMOLS', resgmmols.bse)
        print("Hausman's specification test")
        print(resls.spec_hausman())
        print(spec_hausman(resols.params, res.params, resols.cov_params(), res.cov_params()))
        print(spec_hausman(resgmmols.params, res.params, resgmmols.cov_params(), res.cov_params()))
    if 'distquant' in examples:
        gparrvs = stats.genpareto.rvs(2, size=5000)
        x0p = [1.0, gparrvs.min() - 5, 1]
        moddist = gmm.DistQuantilesGMM(gparrvs, None, None, distfn=stats.genpareto)
        pit1, wit1 = moddist.fititer([1.5, 0, 1.5], maxiter=1)
        print(pit1)
        p1 = moddist.fitgmm([1.5, 0, 1.5])
        print(p1)
        moddist2 = gmm.DistQuantilesGMM(gparrvs, None, None, distfn=stats.genpareto, pquant=np.linspace(0.01, 0.99, 10))
        pit1a, wit1a = moddist2.fititer([1.5, 0, 1.5], maxiter=1)
        print(pit1a)
        p1a = moddist2.fitgmm([1.5, 0, 1.5])
        print(p1a)
        res1b = moddist2.fitonce([1.5, 0, 1.5])
        print(res1b.params)
        print(res1b.bse)
        print(np.sqrt(np.diag(res1b._cov_params)))