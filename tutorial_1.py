# -*- coding: utf-8 -*-


import statistics_lib as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%############################################################################
################################# USER INPUTS #################################
###############################################################################
radius = 3
num_steps = 7

cm_0 = cm.get_cmap('viridis')
cm_1 = cm.get_cmap('plasma')



#%%############################################################################
################################## ANIMATE ####################################
###############################################################################
animate = True

def rotate(angle):
    ax.view_init(azim=angle)

import matplotlib.animation as animation
def animate_3Dplot(figure, start_angle, stop_angle, output_name):
    print('Animating '+output_name+'.gif')
    rot_animation = animation.FuncAnimation(figure, rotate, frames=np.arange(start_angle, stop_angle, 5), interval=250)
    rot_animation.save(output_name+'.gif', dpi=60, writer='imagemagick')

#%%############################################################################
############################## SCRIPT VARIABLES ###############################
###############################################################################
# mean of all distributions
mu = np.array([0,0])

N_contour = radius*100 + 1 # resolutio of contour plots
X_contour = np.linspace( mu[0]-radius, mu[0]+radius, N_contour )
Y_contour = np.linspace( mu[1]-radius, mu[1]+radius, N_contour )
X_contour , Y_contour  = np.meshgrid( X_contour , Y_contour )
contour = np.empty(X_contour.shape + (2,))
contour[:, :, 0] = X_contour
contour[:, :, 1] = Y_contour

plt.close('all')

#%%############################################################################
############################ PLOTTING NORMAL DISTR ############################
###############################################################################
# plot normal distribution
figure = plt.figure(0)
figure.set_size_inches(13.5, 8.5)
ax = figure.add_subplot(projection='3d', proj_type = 'ortho') 
legend_proxy = []

for idx_step, cov in enumerate(np.linspace(-0.9, 0.9, num_steps)):

    Sigma = np.array( [[1, cov], [cov, 1]] )
    
    eig_vals, eig_vecs = np.linalg.eig(Sigma)

    color_0 = cm_0(0.1 + 0.8*((idx_step+1)/(num_steps)) )
    color_1 = cm_1(0.9 - 0.8*((idx_step+1)/(num_steps)) )

    Z_normal = st.multivariate_gaussian(contour, mu, Sigma)
    peak = st.multivariate_gaussian(mu, mu, Sigma)
    ax.contour(  X_contour, Y_contour, Z_normal, 
                    [peak*0.1, peak*0.5, peak*0.99],
                    zdir='z', 
                    offset= [idx_step], 
                    colors= color_0,
                    linestyles= 'dashed')
    
    proxy = plt.Rectangle((0, 0), 1, 1, fc=color_0, label='covariance = '+"{:.2f}".format(cov))
    legend_proxy.append(proxy)
    
    # plot eigenvectors
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,0]*eig_vals[0], eig_vecs[1,0]*eig_vals[0], 0, # <-- directions of vector
        color = 'black', alpha = .8, lw = 2)
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,1]*eig_vals[1], eig_vecs[1,1]*eig_vals[1], 0, # <-- directions of vector
        color = 'black', alpha = .8, lw = 2)
    
proxy = plt.Rectangle((0, 0), 1, 1, fc='black', label='eigen-vector')
legend_proxy.append(proxy)    
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='green', linewidth = 2.0, label='mean')
ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal distribution')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

if(animate):
    animate_3Dplot(figure, 0, 90, 'normal_distr')
#%%############################################################################
######################## PLOTTING NORMAL SKEWED DISTR #########################
###############################################################################
# plot normal skewed distribution
figure = plt.figure(1)
figure.set_size_inches(13.5, 8.5)
ax = figure.add_subplot(projection='3d', proj_type = 'ortho') 
legend_proxy = []
for idx_step, skew in enumerate(np.linspace(-2, 2, num_steps)):

    Sigma = np.array( [[1, 0], [0, 1]] )

    color_0 = cm_0(0.1 + 0.8*((idx_step+1)/(num_steps)) )
    color_2 = cm_0(0.9 - 0.8*((idx_step+1)/(num_steps)) )

    Z_normal = st.multivariate_gaussian_skew(contour, mu, Sigma, np.array([skew, 0]))
    peak = st.multivariate_gaussian(mu, mu, Sigma)
    ax.contour(  X_contour, Y_contour, Z_normal, 
                    [peak*0.1, peak*0.5, peak*0.99],
                    zdir='z', 
                    offset= [idx_step], 
                    colors= color_0,
                    linestyles= 'dashed')
    
    proxy = plt.Rectangle((0, 0), 1, 1, fc=color_0, label='skewness = '+"{:.2f}".format(skew))
    legend_proxy.append(proxy)
    
    # plot eigenvectors
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        skew, 0, 0, # <-- directions of vector
        color = 'gray', alpha = .8, lw = 2)
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='green', linewidth = 2.0, label='mean')
 
proxy = plt.Rectangle((0, 0), 1, 1, fc='gray', label='skewness vector')
legend_proxy.append(proxy)    

ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal skewed distribution')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

if(animate):
    animate_3Dplot(figure, 0, 90, 'skew_distr')
    
#%%############################################################################
################# PLOTTING NORMAL SKEWED DISTR WITH 2 CHNAGES #################
###############################################################################
figure = plt.figure(2)
figure.set_size_inches(13.5, 8.5)
ax = figure.add_subplot(projection='3d', proj_type = 'ortho') 
legend_proxy = []
for idx_step, (skew, cov) in enumerate( zip(np.linspace(-2, 2, num_steps), np.linspace(-0.9, 0.9, num_steps)) ):

    Sigma = np.array( [[1, cov], [cov, 1]] )
    
    eig_vals, eig_vecs = np.linalg.eig(Sigma)
    
    color_0 = cm_0(0.1 + 0.8*((idx_step+1)/(num_steps)) )
    color_1 = cm_1(0.9 - 0.8*((idx_step+1)/(num_steps)) )
    color_2 = cm_0(0.9 - 0.8*((idx_step+1)/(num_steps)) )


    Z_normal = st.multivariate_gaussian_skew(contour, mu, Sigma, np.array([skew, 0]))
    peak = st.multivariate_gaussian(mu, mu, Sigma)
    ax.contour(  X_contour, Y_contour, Z_normal, 
                    [peak*0.1, peak*0.5, peak*0.99],
                    zdir='z', 
                    offset= [idx_step], 
                    colors= color_0,
                    linestyles= 'dashed')
    
    proxy = plt.Rectangle((0, 0), 1, 1, fc=color_0, label='skewness/cov = '+"{:.2f}".format(skew) + ' / ' + "{:.2f}".format(cov) )
    legend_proxy.append(proxy)
    
    # plot eigenvectors
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,0]*eig_vals[0], eig_vecs[1,0]*eig_vals[0], 0, # <-- directions of vector
        color = 'black', alpha = .8, lw = 2)
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,1]*eig_vals[1], eig_vecs[1,1]*eig_vals[1], 0, # <-- directions of vector
        color = 'black', alpha = .8, lw = 2)    
    
    # plot eigenvectors
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        skew, 0, 0, # <-- directions of vector
        color = 'gray', alpha = .8, lw = 2)
    
proxy = plt.Rectangle((0, 0), 1, 1, fc='black', label='eigen-vector')
legend_proxy.append(proxy)    
proxy = plt.Rectangle((0, 0), 1, 1, fc='gray', label='skewness vector')
legend_proxy.append(proxy)        
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='green', linewidth = 2.0, label='mean')
ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal skewed distribution with varying covariance')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

if(animate):
    animate_3Dplot(figure, 0, 90, 'normal_skew_distr')

