# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

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
    
    
def multivariate_gaussian_skew(point, mean, Cov, shape):
    ### check against https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
    
    
    # get the dimension of the problem
    dim = len(mean)    
    # check & assert proper form of inputs
    if(Cov.shape != (dim, dim)):
        raise ValueError('dim(mean)='+str(dim)+', but the dim(Cov)=('+str(Cov.shape[0])+', '+str(Cov.shape[1])+') instead.')
    else:
        Psi  = Cov
    # assert proper dimensions (scipy expect 1-dimensional vectors)
    lamb = shape.reshape(dim,)
    mu   = mean.reshape(dim,)

    # check dimension of the data
    if(len(point.shape)==1):
        point = point.reshape(-1, dim)
    elif(point.shape[1]!=dim):
        point = point.transpose()
    
    # subtract mean
    point = point - mu
    
    # calculate itnernal parameters
    delta = lamb / np.sqrt(1+(lamb**2))
    Delta = np.diagflat( np.sqrt( 1-(delta**2) ) )
    Omega = Delta @ (Psi + np.outer(lamb, lamb.transpose())) @ Delta
    alpha = (lamb.transpose() @ np.linalg.inv(Psi) @ np.linalg.inv(Delta) ) / np.sqrt( 1 + (lamb.transpose() @ np.linalg.inv(Psi) @ lamb ) )

    # evaluate skew normal distribution
    pdf  = multivariate_normal( np.zeros_like(mu), Omega).logpdf( point )
    cdf  = norm(0, 1).logcdf( point  @ alpha )
    probability = np.exp( np.log(2) + pdf + cdf )
    
    return probability
    



class multivariate_normal_skew:
    
    def __init__(self, mean, Cov, shape):
        # get the dimension
        self._dim = len(mean)
        
        # check & assert proper form of inputs
        if(Cov.shape != (self._dim, self._dim)):
            raise ValueError('dim(mean)='+str(self.dim)+', but the dim(Cov)=('+str(Cov.shape[0])+', '+str(Cov.shape[1])+') instead.')
        else:
            self._Psi  = Cov
        # assert proper dimensions (scipy expect 1-dimensional vectors)
        self._lamb = shape.reshape(self._dim,)
        self._mu   = mean.reshape(self._dim,)
        
        # calculate internal parameters
        self.delta = self._lamb / np.sqrt(1+(self._lamb**2))
        self.Delta = np.diagflat( np.sqrt( 1-(self.delta**2) ) )
        self.Omega = self.Delta @ (self._Psi + np.outer(self._lamb, self.lamb.transpose())) @ self.Delta
        self.alpha = (self._lamb.transpose() @ np.linalg.inv(self._Psi) @ np.linalg.inv(self.Delta) ) \
            / np.sqrt( 1 + (self._lamb.transpose() @ np.linalg.inv(self._Psi) @ self._lamb ) )
        
        
    def pdf(self, x):
        return np.exp( self.logpdf(x) )


    def logpdf(self, x):
        # check dimension of x
        if(len(x.shape)==1):
            data = x.reshape(-1,self. _dim)
        elif(x.shape[1]!=self._dim):
            data = x.transpose()
            if(data.shape[1]!=self._dim):
                raise ValueError('"x" doesn\'t have the right dimension even after transposition! The dim(x)='+str(x.shape)+'')
        
        # evaluate skew normal distribution
        data = x - self._mu
        pdf  = multivariate_normal( np.zeros_like(self._mu), self.Omega).logpdf( data )
        cdf  = norm(0, 1).logcdf( data  @ self.alpha )
        log_probability = np.log(2) + pdf + cdf
        return log_probability


    def rvs(self, size=1):
        aCa      = self.alpha @ self.Omega @ self.alpha
        delta    = (1 / np.sqrt(1 + aCa)) * self.Omega @ self.alpha
        cov_star = np.block([[ np.ones(1),       delta ],
                             [      delta,  self.Omega ]])
        x        = multivariate_normal(np.zeros(self._dim+1), cov_star).rvs(size)
        x0, x1   = x[:, 0], x[:, 1:]
        inds     = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1