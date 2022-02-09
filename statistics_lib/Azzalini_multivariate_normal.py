# -*- coding: utf-8 -*-

import numpy as np
from   scipy.stats import multivariate_normal, norm

import matplotlib.pyplot as plt



def multivariate_skew_gaussian(point, mean, Cov, shape):
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

# # sampling from Skew normal distribution
# def rvs_fast(self, mean, Cov, shape, size=1):
    
    
    
#     aCa      = self.shape @ self.cov @ self.shape
#     delta    = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
#     cov_star = np.block([[np.ones(1),     delta],
#                          [delta[:, None], self.cov]])
#     x        = mvn(np.zeros(self.dim+1), cov_star).rvs(size) 
#     x0, x1   = x[:, 0], x[:, 1:]
#     inds     = x0 <= 0
#     x1[inds] = -1 * x1[inds]
#     return x1

# input parameters
x_limits = [-2.0, 2.0]
y_limits = [-2.0, 2.0]

Psi  = np.array([ [0.02, 0.019], [0.019, 0.02] ])
lamb = np.array(  [0.10, 0.0] ).reshape(2,)
mu   = np.array(  [0.3, 0.0] ).reshape(2,)

# re-calculated parameters
delta = lamb / np.sqrt(1+(lamb**2))
Delta = np.diagflat( np.sqrt( 1-(delta**2) ) )

Sigma = Delta @ (Psi + np.outer(lamb, lamb.transpose())) @ Delta

alpha = (lamb.transpose() @ np.linalg.inv(Psi) @ np.linalg.inv(Delta) ) / np.sqrt( 1 + (lamb.transpose() @ np.linalg.inv(Psi) @ lamb ) )


# data
resolution = 100
dx_area = (int((x_limits[1]-x_limits[0])) * int((y_limits[1]-y_limits[0]))) / (int((x_limits[1]-x_limits[0])*resolution+1) * int((y_limits[1]-y_limits[0])*resolution+1))
xx = np.linspace( x_limits[0], x_limits[1], int((x_limits[1]-x_limits[0])*resolution+1) )
yy = np.linspace( y_limits[0], y_limits[1], int((y_limits[1]-y_limits[0])*resolution+1) )
XX , YY  = np.meshgrid( xx , yy )
feature_space = np.empty(XX.shape + (2,))
feature_space[:, :, 0] = XX
feature_space[:, :, 1] = YY

# data = np.array( [XX.flatten(), YY.flatten()] )
data = np.array( [XX.flatten(), YY.flatten()] ).transpose()

# probability
pdf  = multivariate_normal( np.zeros_like(mu), Sigma).logpdf( data - mu )
# cdf  = norm(0, 1).logcdf( data.transpose() @ alpha.transpose() )
# ZZ = np.exp(np.log(2) + pdf + cdf.reshape(-1,))

cdf  = norm(0, 1).logcdf( (data- mu) @ alpha )
ZZ = np.exp(np.log(2) + pdf + cdf )
peak = max(ZZ)


ZZ2 = multivariate_skew_gaussian(data, mu, Psi, lamb)

# plotting
plt.close('all')
figure, ax0 = plt.subplots(nrows=1, ncols=1)
ax0.set_aspect(aspect='equal')
figure.set_size_inches(8.0, 6.5)
ax0.contour(  XX, YY, ZZ.reshape(XX.shape[0], -1).T, 
                np.linspace(0.1*peak, 0.9*peak, 3),
                zdir='z', 
                offset= [0], 
                colors= 'black',
                linestyles= 'dashed')

ax0.contour(  XX, YY, ZZ2.reshape(XX.shape[0], -1).T, 
                np.linspace(0.1*peak, 0.9*peak, 3),
                zdir='z', 
                offset= [0], 
                colors= 'red',
                linestyles= 'dotted')

print('Probability (numerical integration over the whole space) = '+str(sum(ZZ)*dx_area) )
print('Probability (numerical integration over the whole space) = '+str(sum(ZZ2)*dx_area) )