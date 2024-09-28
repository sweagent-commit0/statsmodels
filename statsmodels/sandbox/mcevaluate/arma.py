import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arma_mle import Arma

def mcarma22(niter=10, nsample=1000, ar=None, ma=None, sig=0.5):
    """run Monte Carlo for ARMA(2,2)

    DGP parameters currently hard coded
    also sample size `nsample`

    was not a self contained function, used instances from outer scope
      now corrected

    """
    pass
if __name__ == '__main__':
    ' niter 50, sample size=1000, 2 runs\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.542401   -0.09904305  0.30840599  0.2052473 ]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.54681176 -0.09742921  0.2996297   0.20624258]\n\n\n    niter=50, sample size=200, 3 runs\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.64669489 -0.01134491  0.19972259  0.20634019]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.53141595 -0.10653234  0.32297968  0.20505973]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.50244588 -0.125455    0.33867488  0.19498214]\n\n    niter=50, sample size=100, 5 runs  --> ar1 too low, ma1 too high\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.35715008 -0.23392766  0.48771794  0.21901059]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.3554852  -0.21581914  0.51744748  0.24759245]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.3737861  -0.24665911  0.48031939  0.17274438]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.30015385 -0.27705506  0.56168199  0.21995759]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.35879991 -0.22999604  0.4761953   0.19670835]\n\n    new version, with burnin 1000 in DGP and demean\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.56770228 -0.00076025  0.25621825  0.24492449]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.27598305 -0.2312364   0.57599134  0.23582417]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.38059051 -0.17413628  0.45147109  0.20046776]\n\n    [-0.55 -0.1   0.3   0.2 ]\n    [-0.47789765 -0.08650743  0.3554441   0.24196087]\n    '
    ar = [1.0, -0.55, -0.1]
    ma = [1.0, 0.3, 0.2]
    nsample = 200
    run_mc = True
    if run_mc:
        for sig in [0.1, 0.5, 1.0]:
            import time
            t0 = time.time()
            rt, res_rho, res_bse = mcarma22(niter=100, sig=sig)
            print('\nResults for Monte Carlo')
            print('true')
            print(rt)
            print('nsample =', nsample, 'sigma = ', sig)
            print('elapsed time for Monte Carlo', time.time() - t0)
            print('\nMC of rho versus true')
            mc_summary(res_rho, rt)
            print('\nMC of bse versus zero')
            mc_summary(res_bse)
            print('\nMC of bse versus std')
            mc_summary(res_bse, res_rho.std(0))