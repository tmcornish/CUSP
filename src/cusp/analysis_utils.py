##############################################################################
# Module containing generally useful functions for data analysis.
##############################################################################

import numpy as np
from scipy import special


def round_sigfigs(num, sf):
    '''
    Rounds a number to a given number of significant figures

    Parameters
    ----------
    num: float
        The number to be rounded

    sf: int
        The number of significant figures to which num will be rounded.

    Returns
    -------
    num_rounded: float
        The input number rounded to the desired number of significant figures.
    '''
    if num != 0.:
        i = -int(np.floor(np.log10(abs(num))) - (sf - 1))
        num_rounded = round(num, i)
    else:
        num_rounded = 0.
    return num_rounded


def percentiles_nsig(n):
    '''
    Returns the lower and upper percentiles corresponding to the n-sigma
    bounds for a normal distribution.

    Parameters
    ----------
    n: int
        The multiple of sigma for which the percentiles are to be
        calculated.
    '''
    # Fraction of population within the range [mu-n*sigma, mu+n*sigma]
    f = special.erf(n / np.sqrt(2.))
    # Percentiles
    p_lo = 0.5 * (1. - f) * 100.
    p_hi = 0.5 * (1. + f) * 100.
    return (p_lo, p_hi)


def fit_polynomial_analytic(x, y, cov, degree=1, return_cov=False,
                            return_chi2nu=False, return_ci=False):
    '''
    Uses linear algebra to analytically fit a polynomial of degree n to data.

    Parameters
    ----------
    x: array-like
        Values of the independent variable.

    y: array-like
        Values of the dependent variable.

    cov: array-like
        Covariance matrix of the data.

    degree: int
        Degree of the polynomial to be fitted (default 1).

    return_cov: bool
        Whether to return the covariance matrix of the best-fit coefficients
        (default False).

    return_chi2nu: bool
        Whether to return the reduced chi^2 value.

    return_ci: bool
        Whether to return a 1-sigma confidence interval at each value of x.

    Returns
    -------
    BF: numpy.ndarray
        Best-fit coefficients.

    cov_bf: numpy.ndarray
        Covariance matrix of the best-fit coefficients (only returned if
        return_cov is True).

    chi2_red: float
        Reduced chi^2 value (only returned if return_chi2nu is True).

    CI: numpy.ndarray
        Array containing the bounds of the 1-sigma confidence interval at each
        value of x (only returned if return_ci is True).
    '''

    # Construct an array where each column is x^0, x^1, ... x^n
    n = degree + 1
    A = np.vander(x, n, increasing=True)
    # Invert the data covariance matrix
    icov = np.linalg.inv(cov)

    # Set up the matrices for solving the linear system of equations
    M1 = A.T @ icov @ A
    M2 = A.T @ icov @ y

    # Matrix containing best-fit coefficients in ascending order of degree in x
    BF = np.linalg.inv(M1) @ M2
    to_return = [BF]

    # Covariance matrix for the best-fit coefficients
    cov_bf = np.linalg.inv(M1)

    if return_cov:
        to_return.append(cov_bf)

    if return_chi2nu:
        # Compute reduced chi^2
        M3 = y - A @ BF
        chi2 = M3.T @ icov @ M3
        chi2_red = chi2 / (len(y) - n)
        to_return.append(chi2_red)

    if return_ci:
        # Randomly draw 10000 values from a multivariate gaussian
        bf_rand = np.random.multivariate_normal(BF, cov_bf, size=10000)
        # Compute the corresponding values of the function
        y_rand = (bf_rand @ A.T)
        # Get the ~16th and 84th percentiles
        CI = np.percentile(y_rand, percentiles_nsig(1), axis=0)
        to_return.append(CI)

    return (*to_return,)


def cov_to_corr(cov):
    '''
    Converts a covariance matrix into a correlation matrix.

    Parameters
    ----------
    cov: np.ndarray
        Covariance matrix to be converted.

    Returns
    -------
    corr: np.ndarray
        The corresponding correlation matrix.
    '''
    Dinv = np.diag(1. / np.sqrt(np.diag(cov)))
    corr = Dinv @ cov @ Dinv

    return corr
