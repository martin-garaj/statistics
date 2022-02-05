# -*- coding: utf-8 -*-


import statistics_lib as st
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

plt.close('all')
mu = np.array([0,0])
radius = 3
num_steps = 7

cm_0 = cm.get_cmap('viridis')
cm_1 = cm.get_cmap('plasma')






from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)





# Our 2-dimensional distribution will be over variables X and Y
N_contour = 251
X_contour = np.linspace( mu[0]-radius, mu[0]+radius, N_contour )
Y_contour = np.linspace( mu[1]-radius, mu[1]+radius, N_contour )
X_contour , Y_contour  = np.meshgrid( X_contour , Y_contour )
# Pack X and Y into a single 3-dimensional array
contour = np.empty(X_contour.shape + (2,))
contour[:, :, 0] = X_contour
contour[:, :, 1] = Y_contour


# plot normal distribution
figure = plt.figure(0)
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
                    [peak*0.1, peak*0.5, peak*0.9],
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
        color = color_1, alpha = .8, lw = 2)
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,1]*eig_vals[1], eig_vecs[1,1]*eig_vals[1], 0, # <-- directions of vector
        color = color_1, alpha = .8, lw = 2)
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='black', linewidth = 2.0, label='mean')


ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal distribution')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

plt.show()

# plot normal skewed distribution
figure = plt.figure(1)
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
        color = color_2, alpha = .8, lw = 2)
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='black', linewidth = 2.0, label='mean')
    
ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal skewed distribution')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

plt.show()



# plot normal skewed distribution
figure = plt.figure(2)
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
                    [peak*0.1, peak*0.5, peak*0.9],
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
        color = color_1, alpha = .8, lw = 2)
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        eig_vecs[0,1]*eig_vals[1], eig_vecs[1,1]*eig_vals[1], 0, # <-- directions of vector
        color = color_1, alpha = .8, lw = 2)    
    
    # plot eigenvectors
    ax.quiver(
        0, 0, idx_step, # <-- starting point of vector
        skew, 0, 0, # <-- directions of vector
        color = color_2, alpha = .8, lw = 2)
    
ax.plot([0, 0], [0,0], [-1, num_steps+1], color='black', linewidth = 2.0, label='mean')

ax.set_xlabel('X')
ax.set_xlim(mu[0]-radius, mu[0]+radius)
ax.set_ylabel('Y')
ax.set_ylim(mu[1]-radius, mu[1]+radius)
ax.set_zlabel('Z')
ax.set_zlim(-1, num_steps+1)
ax.set_title('Normal skewed distribution with ')
ax.legend(legend_proxy, [proxy.get_label() for proxy in legend_proxy], bbox_to_anchor=(1.5, 1.1))

plt.show()
