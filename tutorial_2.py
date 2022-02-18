# -*- coding: utf-8 -*-
"""
Tutorial 2

This tutorial illustrates the the decision boundary that arises when the 
feature space is dvided using 2 distribtions.
"""

#%%############################################################################
################################### IMPORTS ###################################
###############################################################################
# import the statistics_lib package that includes the implementation of 
# the distributions
import statistics_lib as st

# nice way to import things
# from   scipy.stats import (multivariate_normal as mvn, norm)

# import other necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#%%############################################################################
################################# USER INPUTS #################################
###############################################################################
# radius ~> xlim and ylim
x_limits = [-0.0, 2.0]
y_limits = [-0.0, 2.0]
sensitivity_limit = 1.0
boost = 1e-1
# select colormaps
cm_0 = cm.get_cmap('seismic')
cm_1 = cm.get_cmap('plasma')

model = { 0: {'mean':np.array([0.25, 1.0]), 'Cov':np.array([ [0.0, 0.0],    [0.0, 0.0]]),    'shape':np.array([0.4, 0.2])},
          1: {'mean':np.array([1.0, 1.0]),  'Cov':np.array([ [0.03, 0.019], [0.019, 0.02]]), 'shape':np.array([-0.4, 0.5])},
          }

# model = { 0: {'mu':np.array([-3.25, 1.0]), 'Sigma':np.array([ [0.0, 0.0],    [0.0, 0.0]]),    'skew':np.array([0.0, 0.0])},
#           1: {'mu':np.array([1.0, 1.0]), 'Sigma':np.array([ [1.5, -0.9], [-0.9, 1.5]]),  'skew':np.array([5.0, 0.0])},
#           }

# rotate covariance matrix Sigma
# replace parking model by a rotated generative model
# this enables to use the probability distributions as 
# distance measure that creates a straight line 
eig_vals, eig_vecs = np.linalg.eig( model[1]['Cov'] )
# pick the major eigen-vector
eig_vec_0 = eig_vecs[ :, eig_vals.argmax() ]
rot_angle_0 = np.arctan2( eig_vec_0[0], eig_vec_0[1] )
# do the same for Sigma_parking
rot_angle_1 = np.arctan2( 0.3, 1.0 )
# find the angle
rot_angle_data = rot_angle_1 - rot_angle_0
# assure the angle is not more than pi/2
rot_angle_data = ( rot_angle_data - np.pi ) if (rot_angle_data >= np.pi/2) else rot_angle_data
rot_angle_data = ( rot_angle_data + np.pi ) if (rot_angle_data < -np.pi/2)  else rot_angle_data          
# print(rot_angle_data)
c, s = np.cos(rot_angle_data), np.sin(rot_angle_data)
R_data = np.array(((c, -s), (s, c)))
# rorate Sigma_generative to align with 
model[0]['Cov'] = R_data @ model[1]['Cov'] @ R_data.T



# if True, then the results are saved as gifs (takes a while so requires patience)
animate = True 

#%%############################################################################
################################## ANIMATE ####################################
###############################################################################


#%%############################################################################
############################## SCRIPT VARIABLES ###############################
###############################################################################
# mean of all distributions

resolution = 100
xx = np.linspace( x_limits[0], x_limits[1], int((x_limits[1]-x_limits[0])*resolution+1) )
yy = np.linspace( y_limits[0], y_limits[1], int((y_limits[1]-y_limits[0])*resolution+1) )
XX , YY  = np.meshgrid( xx , yy )
feature_space = np.empty(XX.shape + (2,))
feature_space[:, :, 0] = XX
feature_space[:, :, 1] = YY

feature_space_flat = np.array( [XX.flatten(), YY.flatten()] ).transpose()

#%%############################################################################
############################ GENERATE IMAGES ############################
###############################################################################
# for idx_mu0, mu0 in enumerate(np.append(np.linspace(0.2, 4.7, 3), np.linspace(4.7, 0.2, 3))):
for idx_mean0, mean0 in enumerate( np.linspace(3.5, -1.5, 30) ):
    model[0]['mean'][1] = mean0
    
    plt.close('all')
    
    #%%############################################################################
    ############################ PLOTTING NORMAL DISTR ############################
    ###############################################################################
    # plot normal distribution
    figure, ax0 = plt.subplots(nrows=1, ncols=1)
    ax0.set_aspect(aspect='equal')
    figure.set_size_inches(8.0, 6.5)
    legend_proxy = []
    
    results = dict()
    ZZs = dict()
    for idx_name, name in enumerate(model):
    
        # get model parameters
        mean = model[name]['mean']
        Cov  = model[name]['Cov']
        shape = model[name]['shape']
        
        mns = st.multivariate_normal_skew(mean, Cov, shape)
    
        # color_0 = cm_0(0.1 + 0.8*((idx_name+1)/(len(model))) )
        # color_1 = cm_0(0.1 + 0.8*((idx_name+1)/(len(model))) )      
    
        ZZ   = mns.pdf(feature_space_flat).reshape(XX.shape[0], -1).T 
        peak = ZZ.max()
        
        # store the output
        ZZs[name] = ZZ/peak
        # results[name]['ZZ'] = ZZ
        # results[name]['peak'] = peak
        
        ax0.contour(  XX, YY, ZZ, 
                        np.linspace(0.1*peak, 0.9*peak, 3),
                        zdir='z', 
                        offset= [0], 
                        colors= 'gray' if name==0 else 'white',
                        linestyles= 'dashed')
        
        # add proxy for legend purposes
        # proxy = plt.Rectangle((0, 0), 1, 1, fc=color_0, label='covariance = '+name)
        # legend_proxy.append(proxy)
        
    # add legend if necessary
    # proxy = plt.Rectangle((0, 0), 1, 1, fc='black', label='eigen-vector')
    # legend_proxy.append(proxy)
        
    #%%############################################################################
    ############################## PROCESS THE DATA ###############################
    ###############################################################################
    ZZ_0_norm = ZZs[0] / ZZs[0].sum()
    ZZ_1_norm = ZZs[1] / ZZs[1].sum()
    Bhattacharyya_coeff = np.sqrt( ZZ_0_norm * ZZ_1_norm ).sum()   
    ax0.text(x_limits[0]+0.1, y_limits[0]+0.1, 'Bhattacharyya coeff. = '+"{:.3f}".format(Bhattacharyya_coeff))
    
    # process the outputs
    ZZdiff = ZZs[0] - ZZs[1]
    
    ZZratio = (ZZs[0]+boost) /  ((ZZs[1]+boost) * sensitivity_limit)
    ZZratio[ZZratio > 1.0] = (2.0 - (1.0/ZZratio))[ZZratio > 1.0]
    
    # sigmoid function
    # ZZratio = 1/(1 + np.exp(-ZZratio / 100000))
    
    ZZproc = ZZratio
    # remove the first and last row
    ZZproc = ZZproc[:-1, :-1]
    # divide the ZZ space into levels
    # levels = MaxNLocator(nbins=250).tick_values(ZZproc.min(), ZZproc.max())
    levels = MaxNLocator(nbins=20).tick_values(0, 2.0)
    # select a colormap
    cmap = cm_0
    # normalize the data
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # plot pcolormesh
    pmesh = plt.pcolormesh(XX, YY, ZZproc, cmap=cmap, norm=norm)
    
    # create a colorbar
    figure.colorbar(pmesh, ax=ax0)
    # set layout
    figure.tight_layout()
    
    # ax.plot([0, 0], [0,0], [-1, num_steps+1], color='green', linewidth = 2.0, label='mean')
    ax0.set_xlabel('X')
    ax0.set_xlim(x_limits[0], x_limits[1])
    ax0.set_ylabel('Y')
    ax0.set_ylim(y_limits[0], y_limits[1])
    # ax.set_zlabel('Z')
    # ax.set_zlim(-1, num_steps+1)
    ax0.set_title('Categorization of feature space')
    # ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))
    
    
    #%%############################################################################
    ############################## CREATE ANIMATION ###############################
    ###############################################################################
    if(animate):
        plt.savefig('./graphics/distribution_overlap/tutorial_2_gif_'+str(idx_mean0)+'.png')
