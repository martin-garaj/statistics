# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import norm

def multivariate_gaussian(point, mu, Sigma):
    """
    Return likelihood of a point belonging within a normal distribution 
    characterized by mean "mu" and covariance matrix "Sigma"

    Parameters
    ----------
    pos : <numpy.array>
        Multi dimensional data point.
    mu : <numpy.array>
        Mean value of the normal distriution.
    Sigma : <numpy.array>
        Covariance matrix of the distribution.

    Returns
    -------
    numpy.array
        Likelihood of the point belonging to the distribution.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', point-mu, Sigma_inv, point-mu)

    return np.exp(-fac / 2) / N

def gaussian_quantile(point, quantile, mu, Sigma):
    """
    Given point(s), the requested quantile and the Gaussian distribution 
    parameters, the function returns True/False on whether the point belongs
    to the given quantile.

    Parameters
    ----------
    point : <nympu.array>
        Data point.
    quantile : <float>
        Requested quantile.
    mu : <numpy.array>
        Mean value of the normal distriution.
    Sigma : <numpy.array>
        Covariance matrix of the distribution.

    Returns
    -------
    bool
        Bool (or array of bools) for every "point" on whether it belongs to 
        the quartile (True) or not (False).

    """
    Sigma_inv = np.linalg.inv(Sigma)
    fac = np.einsum('...k,kl,...l->...', point-mu, Sigma_inv, point-mu)
    
    return ( fac + 2*np.ln(quantile) ) > 0.0
    
    
def multivariate_gaussian_skew(pos, mu, Sigma, skewness):
    """
    Return likelihood of a point belonging within a normal distribution 
    with skewness characterized by mean "mu", covariance matrix "Sigma" 
    and "skewness".

    Parameters
    ----------
    pos : <numpy.array>
        Multi dimensional data point.
    mu : <numpy.array>
        Mean value of the normal distriution.
    Sigma : <numpy.array>
        Covariance matrix of the distribution.
    skewness : <numpy.array>
        Skewness vector with the same dimension as the mean "mu".
    
    Returns
    -------
    numpy.array
        Likelihood of the point belonging to the distribution.

    """    
    
    # get the dimension of multivariete nrmal distribution
    n = mu.shape[0]
    # get determinant of Sigma
    Sigma_det = np.linalg.det(Sigma)
    # evaluate denominator
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # get inverse of Sigma
    Sigma_inv = np.linalg.inv(Sigma)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    # evaluate multivatiate normal
    multivariate = np.exp(-fac / 2) / N
    # evaluate multivariate skewed normal
    skew_normal = 2 * multivariate * norm.cdf( np.inner( pos, skewness ) )
    
    # return
    return skew_normal
    
