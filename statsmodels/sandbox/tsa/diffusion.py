"""getting started with diffusions, continuous time stochastic processes

Author: josef-pktd
License: BSD


References
----------

An Algorithmic Introduction to Numerical Simulation of Stochastic Differential
Equations
Author(s): Desmond J. Higham
Source: SIAM Review, Vol. 43, No. 3 (Sep., 2001), pp. 525-546
Published by: Society for Industrial and Applied Mathematics
Stable URL: http://www.jstor.org/stable/3649798

http://www.sitmo.com/  especially the formula collection


Notes
-----

OU process: use same trick for ARMA with constant (non-zero mean) and drift
some of the processes have easy multivariate extensions

*Open Issues*

include xzero in returned sample or not? currently not

*TODOS*

* Milstein from Higham paper, for which processes does it apply
* Maximum Likelihood estimation
* more statistical properties (useful for tests)
* helper functions for display and MonteCarlo summaries (also for testing/checking)
* more processes for the menagerie (e.g. from empirical papers)
* characteristic functions
* transformations, non-linear e.g. log
* special estimators, e.g. Ait Sahalia, empirical characteristic functions
* fft examples
* check naming of methods, "simulate", "sample", "simexact", ... ?



stochastic volatility models: estimation unclear

finance applications ? option pricing, interest rate models


"""
import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt

class Diffusion:
    """Wiener Process, Brownian Motion with mu=0 and sigma=1
    """

    def __init__(self):
        pass

    def simulateW(self, nobs=100, T=1, dt=None, nrepl=1):
        """generate sample of Wiener Process
        """
        pass

    def expectedsim(self, func, nobs=100, T=1, dt=None, nrepl=1):
        """get expectation of a function of a Wiener Process by simulation

        initially test example from
        """
        pass

class AffineDiffusion(Diffusion):
    """

    differential equation:

    :math::
    dx_t = f(t,x)dt + \\sigma(t,x)dW_t

    integral:

    :math::
    x_T = x_0 + \\int_{0}^{T}f(t,S)dt + \\int_0^T  \\sigma(t,S)dW_t

    TODO: check definition, affine, what about jump diffusion?

    """

    def __init__(self):
        pass

    def simEM(self, xzero=None, nobs=100, T=1, dt=None, nrepl=1, Tratio=4):
        """

        from Higham 2001

        TODO: reverse parameterization to start with final nobs and DT
        TODO: check if I can skip the loop using my way from exactprocess
              problem might be Winc (reshape into 3d and sum)
        TODO: (later) check memory efficiency for large simulations
        """
        pass
'\n    R = 4; Dt = R*dt; L = N/R;        % L EM steps of size Dt = R*dt\n    Xem = zeros(1,L);                 % preallocate for efficiency\n    Xtemp = Xzero;\n    for j = 1:L\n       Winc = sum(dW(R*(j-1)+1:R*j));\n       Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;\n       Xem(j) = Xtemp;\n    end\n'

class ExactDiffusion(AffineDiffusion):
    """Diffusion that has an exact integral representation

    this is currently mainly for geometric, log processes

    """

    def __init__(self):
        pass

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """ddt : discrete delta t



        should be the same as an AR(1)
        not tested yet
        """
        pass

class ArithmeticBrownian(AffineDiffusion):
    """
    :math::
    dx_t &= \\mu dt + \\sigma dW_t
    """

    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

    def exactprocess(self, nobs, xzero=None, ddt=1.0, nrepl=2):
        """ddt : discrete delta t

        not tested yet
        """
        pass

class GeometricBrownian(AffineDiffusion):
    """Geometric Brownian Motion

    :math::
    dx_t &= \\mu x_t dt + \\sigma x_t dW_t

    $x_t $ stochastic process of Geometric Brownian motion,
    $\\mu $ is the drift,
    $\\sigma $ is the Volatility,
    $W$ is the Wiener process (Brownian motion).

    """

    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

class OUprocess(AffineDiffusion):
    """Ornstein-Uhlenbeck

    :math::
      dx_t&=\\lambda(\\mu - x_t)dt+\\sigma dW_t

    mean reverting process



    TODO: move exact higher up in class hierarchy
    """

    def __init__(self, xzero, mu, lambd, sigma):
        self.xzero = xzero
        self.lambd = lambd
        self.mu = mu
        self.sigma = sigma

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """ddt : discrete delta t

        should be the same as an AR(1)
        not tested yet
        # after writing this I saw the same use of lfilter in sitmo
        """
        pass

    def fitls(self, data, dt):
        """assumes data is 1d, univariate time series
        formula from sitmo
        """
        pass

class SchwartzOne(ExactDiffusion):
    """the Schwartz type 1 stochastic process

    :math::
    dx_t = \\kappa (\\mu - \\ln x_t) x_t dt + \\sigma x_tdW \\

    The Schwartz type 1 process is a log of the Ornstein-Uhlenbeck stochastic
    process.

    """

    def __init__(self, xzero, mu, kappa, sigma):
        self.xzero = xzero
        self.mu = mu
        self.kappa = kappa
        self.lambd = kappa
        self.sigma = sigma

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """uses exact solution for log of process
        """
        pass

    def fitls(self, data, dt):
        """assumes data is 1d, univariate time series
        formula from sitmo
        """
        pass

class BrownianBridge:

    def __init__(self):
        pass

class CompoundPoisson:
    """nobs iid compound poisson distributions, not a process in time
    """

    def __init__(self, lambd, randfn=np.random.normal):
        if len(lambd) != len(randfn):
            raise ValueError('lambd and randfn need to have the same number of elements')
        self.nobj = len(lambd)
        self.randfn = randfn
        self.lambd = np.asarray(lambd)
"\nrandn('state',100)                                % set the state of randn\nT = 1; N = 500; dt = T/N; t = [dt:dt:1];\n\nM = 1000;                                         % M paths simultaneously\ndW = sqrt(dt)*randn(M,N);                         % increments\nW = cumsum(dW,2);                                 % cumulative sum\nU = exp(repmat(t,[M 1]) + 0.5*W);\nUmean = mean(U);\nplot([0,t],[1,Umean],'b-'), hold on               % plot mean over M paths\nplot([0,t],[ones(5,1),U(1:5,:)],'r--'), hold off  % plot 5 individual paths\nxlabel('t','FontSize',16)\nylabel('U(t)','FontSize',16,'Rotation',0,'HorizontalAlignment','right')\nlegend('mean of 1000 paths','5 individual paths',2)\n\naverr = norm((Umean - exp(9*t/8)),'inf')          % sample error\n"
if __name__ == '__main__':
    doplot = 1
    nrepl = 1000
    examples = []
    if 'all' in examples:
        w = Diffusion()
        ws = w.simulateW(1000, nrepl=nrepl)
        if doplot:
            plt.figure()
            tmp = plt.plot(ws[0].T)
            tmp = plt.plot(ws[0].mean(0), linewidth=2)
            plt.title('Standard Brownian Motion (Wiener Process)')
        func = lambda t, W: np.exp(t + 0.5 * W)
        us = w.expectedsim(func, nobs=500, nrepl=nrepl)
        if doplot:
            plt.figure()
            tmp = plt.plot(us[0].T)
            tmp = plt.plot(us[1], linewidth=2)
            plt.title('Brownian Motion - exp')
        averr = np.linalg.norm(us[1] - np.exp(9 * us[2] / 8.0), np.inf)
        print(averr)
        gb = GeometricBrownian(xzero=1.0, mu=0.01, sigma=0.5)
        gbs = gb.simEM(nobs=100, nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(gbs.T)
            tmp = plt.plot(gbs.mean(0), linewidth=2)
            plt.title('Geometric Brownian')
            plt.figure()
            tmp = plt.plot(np.log(gbs).T)
            tmp = plt.plot(np.log(gbs.mean(0)), linewidth=2)
            plt.title('Geometric Brownian - log-transformed')
        ab = ArithmeticBrownian(xzero=1, mu=0.05, sigma=1)
        abs = ab.simEM(nobs=100, nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(abs.T)
            tmp = plt.plot(abs.mean(0), linewidth=2)
            plt.title('Arithmetic Brownian')
        ou = OUprocess(xzero=2, mu=1, lambd=0.5, sigma=0.1)
        ous = ou.simEM()
        oue = ou.exact(1, 1, np.random.normal(size=(5, 10)))
        ou.exact(0, np.linspace(0, 10, 10 / 0.1), 0)
        ou.exactprocess(0, 10)
        print(ou.exactprocess(0, 10, ddt=0.1, nrepl=10).mean(0))
        oues = ou.exactprocess(0, 100, ddt=0.1, nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(oues.T)
            tmp = plt.plot(oues.mean(0), linewidth=2)
            plt.title('Ornstein-Uhlenbeck')
        so = SchwartzOne(xzero=0, mu=1, kappa=0.5, sigma=0.1)
        sos = so.exactprocess(0, 50, ddt=0.1, nrepl=100)
        print(sos.mean(0))
        print(np.log(sos.mean(0)))
        doplot = 1
        if doplot:
            plt.figure()
            tmp = plt.plot(sos.T)
            tmp = plt.plot(sos.mean(0), linewidth=2)
            plt.title('Schwartz One')
        print(so.fitls(sos[0, :], dt=0.1))
        sos2 = so.exactprocess(0, 500, ddt=0.1, nrepl=5)
        print('true: mu=1, kappa=0.5, sigma=0.1')
        for i in range(5):
            print(so.fitls(sos2[i], dt=0.1))
        bb = BrownianBridge()
        bbs, t, wm = bb.simulate(0, 0.5, 99, nrepl=500, ddt=1.0, sigma=0.1)
        if doplot:
            plt.figure()
            tmp = plt.plot(bbs.T)
            tmp = plt.plot(bbs.mean(0), linewidth=2)
            plt.title('Brownian Bridge')
            plt.figure()
            plt.plot(wm, 'r', label='theoretical')
            plt.plot(bbs.std(0), label='simulated')
            plt.title('Brownian Bridge - Variance')
            plt.legend()
    cp = CompoundPoisson([1, 1], [np.random.normal, np.random.normal])
    cps = cp.simulate(nobs=20000, nrepl=3)
    print(cps[0].sum(-1).sum(-1))
    print(cps[0].sum())
    print(cps[0].mean(-1).mean(-1))
    print(cps[0].mean())
    print(cps[1].size)
    print(cps[1].sum())