# -*- coding: utf-8 -*-

#%%############################################################################
################################### IMPORTS ###################################
###############################################################################
# import the statistics_lib package that includes the implementation of 
# the distributions
import statistics_lib as st
from scipy.stats import norm
from   scipy.stats import multivariate_normal

# import other necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


#%%############################################################################
################################# USER INPUTS #################################
###############################################################################

num_steps = 100

# radius ~> xlim and ylim
x_limits = [-5.1, 5.1]
y_limits = [-5.1, 5.1]

# select colormaps
cm_0 = cm.get_cmap('plasma')

mean  = np.array(  [0.0, 0.0]  )
# Cov   = np.array( [[0.002, 0.019], [0.019, 30.0]] )
Cov   = np.array( [[3.0, 0.3], [0.3, 10.0]] )
shape = np.array(  [5.0, 1.0]  )

# if True, then the results are saved as gifs (takes a while so requires patience)
animate = True 

#%%############################################################################
################################# FUNCTIONS ###################################
###############################################################################



#%%############################################################################
############################## SCRIPT VARIABLES ###############################
###############################################################################
# mean of all distributions

resolution = 10
xx = np.linspace( x_limits[0], x_limits[1], int((x_limits[1]-x_limits[0])*resolution+1) )
yy = np.linspace( y_limits[0], y_limits[1], int((y_limits[1]-y_limits[0])*resolution+1) )
XX , YY  = np.meshgrid( xx , yy )
feature_space = np.empty(XX.shape + (2,))
feature_space[:, :, 0] = XX
feature_space[:, :, 1] = YY

#%%############################################################################
############################ GENERATE IMAGES ############################
###############################################################################
plt.close('all')

#%%############################################################################
############################ PLOTTING NORMAL DISTR ############################
###############################################################################
# plot normal distribution
figure, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(8.0, 6.5)
legend_proxy = []

data = np.array( [XX.flatten(), YY.flatten()] ).transpose()
# ZZ = st.multivariate_gaussian_skew( data, mean, Cov, shape ).reshape(XX.shape[0], -1).T

point = data
###############################################################################
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
###############################################################################
ZZ = probability.reshape(XX.shape[0], -1).T
peak = ZZ.max()





DD_dx = np.zeros_like(data)

for idx_x, x in enumerate(data):
    # d_dx = -(1/5)*2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
    #     * ( ( norm(0, 1).cdf( x  @ alpha ) * ( -np.linalg.inv(Omega) @ x)  ) +  ( (norm(0, 1).pdf( x  @ alpha )**2) *( ( -x  @ alpha) * (-alpha) ) ) )
    
    d_dx = 2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
        * ( ( ( -np.linalg.inv(Omega) @ x ) * norm(0, 1).cdf( alpha @ x ) )  + ( (norm(0, 1).pdf( x  @ alpha )*alpha ) ) )
    
    # d_dx = 2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
    #     * ( ( ( -np.linalg.inv(Omega) @ x ) * norm(0, 1).cdf( alpha @ x ) )  + ( (norm(0, 1).pdf( x  @ alpha )*alpha ) )        
    
    DD_dx[idx_x] = d_dx
    ###############################################################################
DD = DD_dx.reshape(XX.shape[0], XX.shape[1], 2).T



cset0 = ax0.contourf(XX, YY, DD[0,:,:], zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)
cset1 = ax1.contourf(XX, YY, DD[1,:,:], zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)
norm_data = np.log( np.linalg.norm( DD, axis=0 )+1 )
cset2 = ax2.contourf(XX, YY, norm_data, np.linspace(norm_data.min(), norm_data.max(), 40), zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)

ax0.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.9*peak, 3),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')
ax1.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.9*peak, 3),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')

ax2.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.99999*peak, 10),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')
# # plot eigenvectors
# eig_vals, eig_vecs = np.linalg.eig(Omega)

# ax0.quiver(
#     mu[1], mu[0], # <-- starting point of vector
#     eig_vecs[1,0]*(eig_vals[0]), eig_vecs[0,0]*(eig_vals[0]), # <-- directions of vector
#     color = 'black', alpha = 1.0, lw = 2)
# ax0.quiver(
#     mu[1], mu[0], # <-- starting point of vector
#     eig_vecs[1,1]*(eig_vals[1]), eig_vecs[0,1]*(eig_vals[1]), # <-- directions of vector
#     color = 'black', alpha = 1.0, lw = 2)

ax0.set_aspect(aspect='equal')
ax0.set_xlabel('X')
ax0.set_xlim(x_limits[0], x_limits[1])
ax0.set_ylabel('Y')
ax0.set_ylim(y_limits[0], y_limits[1])
ax0.set_title('Derivative in X-direction')
figure.colorbar(cset0, ax=ax0)

ax1.set_aspect(aspect='equal')
ax1.set_xlabel('X')
ax1.set_xlim(x_limits[0], x_limits[1])
ax1.set_ylabel('Y')
ax1.set_ylim(y_limits[0], y_limits[1])
ax1.set_title('Derivative in Y-direction')
figure.colorbar(cset1, ax=ax1)

ax2.set_aspect(aspect='equal')
ax2.set_xlabel('X')
ax2.set_xlim(x_limits[0], x_limits[1])
ax2.set_ylabel('Y')
ax2.set_ylim(y_limits[0], y_limits[1])
ax2.set_title('Magnitude of the gradient')
figure.colorbar(cset2, ax=ax2)

#%%############################################################################
################################# PEAK SEARCH #################################
###############################################################################


# assure I have all the variables
###############################################################################
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
# pdf  = multivariate_normal( np.zeros_like(mu), Omega).logpdf( point )
# cdf  = norm(0, 1).logcdf( point  @ alpha )
# probability = np.exp( np.log(2) + pdf + cdf )
###############################################################################

#### initial guess at x (with zero-mean)

x = mean + 0.1
step_coeff = 1.0
dx_prev = 0.0
dx_new = 0.0

eig_val_velocity = np.linalg.norm( np.linalg.eig(Omega)[0] )**2

for idx_step, step in enumerate(np.linspace(0,num_steps-1, num_steps)):
    # plot the contours of the distribution
    color_0 = cm_0( (step+1)/num_steps )
    
    pdf  = multivariate_normal( np.zeros_like(mu), Omega).logpdf( x )
    cdf  = norm(0, 1).logcdf( x  @ alpha )
    probability = np.exp( np.log(2) + pdf + cdf )    
    
    # d = 1.0 / eig_val_velocity
    d = 2.0
    
    d_dx = 2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
        * ( ( ( -np.linalg.inv(Omega) @ x ) * norm(0, 1).cdf( alpha @ x ) )  + ( (norm(0, 1).pdf( x  @ alpha )*alpha ) ) )    
      
    dx_new = d * d_dx * step_coeff
    
    if idx_step > 0:
        if( ((np.sign(dx_new)-np.sign(dx_prev)) < 0.0).any() ):
            step_coeff = step_coeff/2.0
    dx_prev = dx_new
    
    dx = dx_new

    # d_dx = -2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) * ( -np.linalg.inv(Omega) @ x) * norm(0, 1).cdf( alpha @ x )
    
    # d_dx = -(1/5)*2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
    #     * ( ( norm(0, 1).cdf( x  @ alpha ) * ( -np.linalg.inv(Omega) @ x) ) +  ( (norm(0, 1).pdf( x  @ alpha )**2) *( ( -x  @ alpha)  ) ) )

    # d_dx = -2*multivariate_normal( np.zeros_like(mu), Omega).pdf(x) \
    #     * ( ( norm(0, 1).cdf( x  @ alpha ) * ( -np.linalg.inv(Omega) @ x) * (-np.linalg.inv(Omega) @ np.ones_like(x))) +  ( norm(0, 1).pdf( x  @ alpha )*(( -x  @ alpha) * alpha ) ) )


    # print(dx)
    # progress teh search
    x_t = x + dx
    
    # plot the step
    ax2.plot( [x_t[0] ,x[0]] + mean[0], [x_t[1] ,x[1]] + mean[1], color=color_0, marker='.')
    
    # plt.plot( [x_t[1] ,x[1]] + mean[1], [x_t[0] ,x[0]] + mean[0], color=color_0, marker='.')
    # close the loop
    x = x_t




def rvs_fast(mean, Omega, alpha, size=1):
    dim = len(mean)
    aCa      = alpha @ Omega @ alpha
    delta    = (1 / np.sqrt(1 + aCa)) * Omega @ alpha
    cov_star = np.block([[np.ones(1),     delta],
                         [delta[:, None], Omega]])
    x        = multivariate_normal(np.zeros(dim+1), cov_star).rvs(size)
    x0, x1   = x[:, 0], x[:, 1:]
    inds     = x0 <= 0
    x1[inds] = -1 * x1[inds]
    return x1

samples = rvs_fast(mean, Omega, alpha, size=1000000)
ax3.hist2d(samples[:,1], samples[:,0], bins=(1000, 1000), cmap=plt.cm.jet, range=[x_limits, y_limits])

ax3.set_aspect(aspect='equal')
ax3.set_xlabel('X')
ax3.set_xlim(x_limits[0], x_limits[1])
ax3.set_ylabel('Y')
ax3.set_ylim(y_limits[0], y_limits[1])
ax3.set_title('Random sampling')
