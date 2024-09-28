"""
Created on Mon Dec 14 19:53:25 2009

Author: josef-pktd

generate arma sample using fft with all the lfilter it looks slow
to get the ma representation first

apply arma filter (in ar representation) to time series to get white noise
but seems slow to be useful for fast estimation for nobs=10000

change/check: instead of using marep, use fft-transform of ar and ma
    separately, use ratio check theory is correct and example works
    DONE : feels much faster than lfilter
    -> use for estimation of ARMA
    -> use pade (scipy.interpolate) approximation to get starting polynomial
       from autocorrelation (is autocorrelation of AR(p) related to marep?)
       check if pade is fast, not for larger arrays ?
       maybe pade does not do the right thing for this, not tried yet
       scipy.pade([ 1.    ,  0.6,  0.25, 0.125, 0.0625, 0.1],2)
       raises LinAlgError: singular matrix
       also does not have roots inside unit circle ??
    -> even without initialization, it might be fast for estimation
    -> how do I enforce stationarity and invertibility,
       need helper function

get function drop imag if close to zero from numpy/scipy source, where?

"""
import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess

class ArmaFft(ArmaProcess):
    """fft tools for arma processes

    This class contains several methods that are providing the same or similar
    returns to try out and test different implementations.

    Notes
    -----
    TODO:
    check whether we do not want to fix maxlags, and create new instance if
    maxlag changes. usage for different lengths of timeseries ?
    or fix frequency and length for fft

    check default frequencies w, terminology norw  n_or_w

    some ffts are currently done without padding with zeros

    returns for spectral density methods needs checking, is it always the power
    spectrum hw*hw.conj()

    normalization of the power spectrum, spectral density: not checked yet, for
    example no variance of underlying process is used

    """

    def __init__(self, ar, ma, n):
        super(ArmaFft, self).__init__(ar, ma)
        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        self.nobs = n
        self.arpoly = np.polynomial.Polynomial(ar)
        self.mapoly = np.polynomial.Polynomial(ma)
        self.nar = len(ar)
        self.nma = len(ma)

    def padarr(self, arr, maxlag, atend=True):
        """pad 1d array with zeros at end to have length maxlag
        function that is a method, no self used

        Parameters
        ----------
        arr : array_like, 1d
            array that will be padded with zeros
        maxlag : int
            length of array after padding
        atend : bool
            If True (default), then the zeros are added to the end, otherwise
            to the front of the array

        Returns
        -------
        arrp : ndarray
            zero-padded array

        Notes
        -----
        This is mainly written to extend coefficient arrays for the lag-polynomials.
        It returns a copy.

        """
        pass

    def pad(self, maxlag):
        """construct AR and MA polynomials that are zero-padded to a common length

        Parameters
        ----------
        maxlag : int
            new length of lag-polynomials

        Returns
        -------
        ar : ndarray
            extended AR polynomial coefficients
        ma : ndarray
            extended AR polynomial coefficients

        """
        pass

    def fftar(self, n=None):
        """Fourier transform of AR polynomial, zero-padded at end to n

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftar : ndarray
            fft of zero-padded ar polynomial
        """
        pass

    def fftma(self, n):
        """Fourier transform of MA polynomial, zero-padded at end to n

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftar : ndarray
            fft of zero-padded ar polynomial
        """
        pass

    def fftarma(self, n=None):
        """Fourier transform of ARMA polynomial, zero-padded at end to n

        The Fourier transform of the ARMA process is calculated as the ratio
        of the fft of the MA polynomial divided by the fft of the AR polynomial.

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftarma : ndarray
            fft of zero-padded arma polynomial
        """
        pass

    def spd(self, npos):
        """raw spectral density, returns Fourier transform

        n is number of points in positive spectrum, the actual number of points
        is twice as large. different from other spd methods with fft
        """
        pass

    def spdshift(self, n):
        """power spectral density using fftshift

        currently returns two-sided according to fft frequencies, use first half
        """
        pass

    def spddirect(self, n):
        """power spectral density using padding to length n done by fft

        currently returns two-sided according to fft frequencies, use first half
        """
        pass

    def _spddirect2(self, n):
        """this looks bad, maybe with an fftshift
        """
        pass

    def spdroots(self, w):
        """spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)
        """
        pass

    def _spdroots(self, arroots, maroots, w):
        """spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)

        Parameters
        ----------
        arroots : ndarray
            roots of ar (denominator) lag-polynomial
        maroots : ndarray
            roots of ma (numerator) lag-polynomial
        w : array_like
            frequencies for which spd is calculated

        Notes
        -----
        this should go into a function
        """
        pass

    def spdpoly(self, w, nma=50):
        """spectral density from MA polynomial representation for ARMA process

        References
        ----------
        Cochrane, section 8.3.3
        """
        pass

    def filter(self, x):
        """
        filter a timeseries with the ARMA filter

        padding with zero is missing, in example I needed the padding to get
        initial conditions identical to direct filter

        Initial filtered observations differ from filter2 and signal.lfilter, but
        at end they are the same.

        See Also
        --------
        tsa.filters.fftconvolve

        """
        pass

    def filter2(self, x, pad=0):
        """filter a time series using fftconvolve3 with ARMA filter

        padding of x currently works only if x is 1d
        in example it produces same observations at beginning as lfilter even
        without padding.

        TODO: this returns 1 additional observation at the end
        """
        pass

    def acf2spdfreq(self, acovf, nfreq=100, w=None):
        """
        not really a method
        just for comparison, not efficient for large n or long acf

        this is also similarly use in tsa.stattools.periodogram with window
        """
        pass

    def invpowerspd(self, n):
        """autocovariance from spectral density

        scaling is correct, but n needs to be large for numerical accuracy
        maybe padding with zero in fft would be faster
        without slicing it returns 2-sided autocovariance with fftshift

        >>> ArmaFft([1, -0.5], [1., 0.4], 40).invpowerspd(2**8)[:10]
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        >>> ArmaFft([1, -0.5], [1., 0.4], 40).acovf(10)
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        """
        pass

    def spdmapoly(self, w, twosided=False):
        """ma only, need division for ar, use LagPolynomial
        """
        pass

    def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
        """Plot results"""
        pass
if __name__ == '__main__':
    nobs = 200
    ar = [1, 0.0]
    ma = [1, 0.0]
    ar2 = np.zeros(nobs)
    ar2[:2] = [1, -0.9]
    uni = np.zeros(nobs)
    uni[0] = 1.0
    arcomb = np.convolve(ar, ar2, mode='same')
    marep = signal.lfilter(ma, arcomb, uni)
    print(marep[:10])
    mafr = fft.fft(marep)
    rvs = np.random.normal(size=nobs)
    datafr = fft.fft(rvs)
    y = fft.ifft(mafr * datafr)
    print(np.corrcoef(np.c_[y[2:], y[1:-1], y[:-2]], rowvar=0))
    arrep = signal.lfilter([1], marep, uni)
    print(arrep[:20])
    arfr = fft.fft(arrep)
    yfr = fft.fft(y)
    x = fft.ifft(arfr * yfr).real
    print(x[:5])
    print(rvs[:5])
    print(np.corrcoef(np.c_[x[2:], x[1:-1], x[:-2]], rowvar=0))
    arcombp = np.zeros(nobs)
    arcombp[:len(arcomb)] = arcomb
    map_ = np.zeros(nobs)
    map_[:len(ma)] = ma
    ar0fr = fft.fft(arcombp)
    ma0fr = fft.fft(map_)
    y2 = fft.ifft(ma0fr / ar0fr * datafr)
    print(y2[:10])
    print(y[:10])
    print(maxabs(y, y2))
    ar = [1, -0.4]
    ma = [1, 0.2]
    arma1 = ArmaFft([1, -0.5, 0, 0, 0, 0, -0.7, 0.3], [1, 0.8], nobs)
    nfreq = nobs
    w = np.linspace(0, np.pi, nfreq)
    w2 = np.linspace(0, 2 * np.pi, nfreq)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    spd1, w1 = arma1.spd(2 ** 10)
    print(spd1.shape)
    _ = plt.plot(spd1)
    plt.title('spd fft complex')
    plt.figure()
    spd2, w2 = arma1.spdshift(2 ** 10)
    print(spd2.shape)
    _ = plt.plot(w2, spd2)
    plt.title('spd fft shift')
    plt.figure()
    spd3, w3 = arma1.spddirect(2 ** 10)
    print(spd3.shape)
    _ = plt.plot(w3, spd3)
    plt.title('spd fft direct')
    plt.figure()
    spd3b = arma1._spddirect2(2 ** 10)
    print(spd3b.shape)
    _ = plt.plot(spd3b)
    plt.title('spd fft direct mirrored')
    plt.figure()
    spdr, wr = arma1.spdroots(w)
    print(spdr.shape)
    plt.plot(w, spdr)
    plt.title('spd from roots')
    plt.figure()
    spdar1_ = spdar1(arma1.ar, w)
    print(spdar1_.shape)
    _ = plt.plot(w, spdar1_)
    plt.title('spd ar1')
    plt.figure()
    wper, spdper = arma1.periodogram(nfreq)
    print(spdper.shape)
    _ = plt.plot(w, spdper)
    plt.title('periodogram')
    startup = 1000
    rvs = arma1.generate_sample(startup + 10000)[startup:]
    import matplotlib.mlab as mlb
    plt.figure()
    sdm, wm = mlb.psd(x)
    print('sdm.shape', sdm.shape)
    sdm = sdm.ravel()
    plt.plot(wm, sdm)
    plt.title('matplotlib')
    from nitime.algorithms import LD_AR_est
    wnt, spdnt = LD_AR_est(rvs, 10, 512)
    plt.figure()
    print('spdnt.shape', spdnt.shape)
    _ = plt.plot(spdnt.ravel())
    print(spdnt[:10])
    plt.title('nitime')
    fig = plt.figure()
    arma1.plot4(fig)