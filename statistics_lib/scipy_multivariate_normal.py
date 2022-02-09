# -*- coding: utf-8 -*-

"""
This script examines the implementation of Normal distribution and Skew Normal
distribution as introduced in https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
"""

import numpy as np
from   scipy.stats import (multivariate_normal as mvn,
                           norm)
from   scipy.stats._multivariate import _squeeze_output

class multivariate_skewnorm:
    
    def __init__(self, shape, cov=None):
        """
        Initial parameter of the Skew Normal distribution.
        
        The mean is assumed as 0-vector.

        Parameters
        ----------
        shape : <np.array>
            Skewness vector of the distribution.
        cov : <np.array>, optional
            Symmetric covariance matrix. The default is None.

        Returns
        -------
        None.
        """
        self.dim   = len(shape)
        self.shape = np.asarray(shape)
        self.mean  = np.zeros(self.dim)
        self.cov   = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        """
        Probability Density Function of the Skew Normal distribution.

        Parameters
        ----------
        x : <np.array>
            Data.

        Returns
        -------
        <np.array>
            Likelihood.
        """
        return np.exp(self.logpdf(x))
        
    def logpdf(self, x):
        """
        Brief notes on some functions:
            _process_quantiles()
                Adjust quantiles array so that last axis labels the 
                components of each data point.
                But thechnically this just adds an extra dimension.
            _squeeze_output()
                Remove single-dimensional entries from array and convert 
                to scalar, if necessary. 
                This is the opposite of _process_quantiles().
            logpdf() and logcdf() 
                These return the log(pdf(x)) and log(cdf(x)) respectively and 
                as such are used to prevent underflow. The results from pdf(x)
                and cdf(x) are obtained as exp(sum(log(pdf(x)))) and
                exp(sum(log(cdf(x)))) respectively.
            
        Parameters
        ----------
        x : <np.array>
            Data.

        Returns
        -------
        <np.array>
            Returns log(pdf(x)) of skew normal distribution.
        """
        x    = mvn._process_quantiles(x, self.dim)
        pdf  = mvn(self.mean, self.cov).logpdf(x)
        cdf  = norm(0, 1).logcdf(np.dot(x, self.shape))
        return _squeeze_output(np.log(2) + pdf + cdf)

    # sampling from Skew normal distribution
    def rvs_fast(self, size=1):
        aCa      = self.shape @ self.cov @ self.shape
        delta    = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block([[np.ones(1),     delta],
                             [delta[:, None], self.cov]])
        x        = mvn(np.zeros(self.dim+1), cov_star).rvs(size) 
        x0, x1   = x[:, 0], x[:, 1:]
        inds     = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1

    def mode(self, precision=1e-4):
        """
        Get the mode position of the Skew Normal distribution distribution.

        Parameters
        ----------
        precision : TYPE, optional
            DESCRIPTION. The default is 1e-4.

        Returns
        -------
        None.

        """
        pass


