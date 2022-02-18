# -*- coding: utf-8 -*-

#%%############################################################################
################################### IMPORTS ###################################
###############################################################################
# import the statistics_lib package that includes the implementation of 
# the distributions
import statistics_lib as st

# import other necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%############################################################################
################################# USER INPUTS #################################
###############################################################################

num_steps = 1000

# radius ~> xlim and ylim
x_limits = [-6.1, 6.1]
y_limits = [-6.1, 6.1]
resolution = 10

# select colormaps
cm_0 = cm.get_cmap('plasma')


mean  = np.array(  [0.0, 0.0]  )
# Cov   = np.array( [[0.002, 0.019], [0.019, 30.0]] )
Cov   = np.array( [[3.0, 0.3], [0.3, 10.0]] )
shape = np.array(  [5.0, 1.0]  )

#%%############################################################################
################################# FUNCTIONS ###################################
###############################################################################
# instantiate class of multivariate normal skewed distribution
mns = st.multivariate_normal_skew(mean, Cov, shape)


#%%############################################################################
############################## SCRIPT VARIABLES ###############################
###############################################################################
# mean of all distributions

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

# evaluate 
ZZ = mns.pdf(data).reshape(XX.shape[0], -1).T
peak = ZZ.max()



DD_dx = mns.get_dx(data)

DD = DD_dx.reshape(XX.shape[0], XX.shape[1], 2).T

cset0 = ax0.contourf(XX, YY, DD[1,:,:], zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)
cset1 = ax1.contourf(XX, YY, DD[0,:,:], zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)
norm_data = np.log( np.linalg.norm( DD, axis=0 )+1 )
cset2 = ax2.contourf(XX, YY, norm_data, np.linspace(norm_data.min(), norm_data.max(), 40), zdir='z', offset=0, cmap=cm.coolwarm, alpha= 0.5)

ax0.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.9*peak, 7),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')
ax1.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.9*peak, 7),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')

ax2.contour(XX, YY, ZZ,
            np.linspace(0.1*peak, 0.9*peak, 7),
            zdir='z',
            offset= [0],
            colors= 'gray',
            linestyles= 'dashed')

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

### TODO:
#   Plot the mode search including the steps taken from initial position until 
#   the mode is reached

#%%############################################################################
########################## RANDOMLY SAMPLED VARIABLE ##########################
###############################################################################
samples = mns.rvs(size=1000000)
ax3.hist2d(samples[:,1], samples[:,0], bins=(1000, 1000), cmap=plt.cm.jet, range=[x_limits, y_limits])

ax3.set_aspect(aspect='equal')
ax3.set_xlabel('X')
ax3.set_xlim(x_limits[0], x_limits[1])
ax3.set_ylabel('Y')
ax3.set_ylim(y_limits[0], y_limits[1])
ax3.set_title('Random sampling')
