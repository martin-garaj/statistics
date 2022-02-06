# -*- coding: utf-8 -*-

import numpy as np



def scatter_hist(x, y, ax, ax_histx, ax_histy, ax_settings=None, histx_settings=None, histy_settings=None, label=''):
    """
    Plots a scatter-histogram with 2D data display and histograms 
    on adjacent axis.

    # Exemplary
    figure = plt.figure(figsize=(8,8), dpi= 100)
    
    # Scatter + Histograms
    gs = figure.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    
    ax       = figure.add_subplot(gs[1, 0])
    ax_histx = figure.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = figure.add_subplot(gs[1, 1], sharey=ax)


    Parameters
    ----------
    x : <np.array>
        X data.
    y : <np.array>
        Y data.
    ax : <axis>
        Handle to the axis with 2D data.
    ax_histx : <axis>
        Handle to the axis with histogram on X axis (top).
    ax_histy : <axis>
        Handle to the axis with histogram on Y axis (right).
    ax_settings : <dict>, optional
        Dictionary with plot settings for ax. The default is None.
    histx_settings : <dict>, optional
        Dictionary with plot settings for ax_histx. The default is None.
    histy_settings : <dict>, optional
        Dictionary with plot settings for ax_histy. The default is None.
    label : <str>, optional
        String for the title of the 2D plot. The default is ''.

    Returns
    -------
    None.
    """
    
    if(isinstance(ax_settings, type(None))):
        ax_settings = dict()
        ax_settings['marker']       = '.'
        ax_settings['color']        = ['black']
        ax_settings['colormap']     = 'viridis'
        ax_settings['alpha']        = 0.5
        ax_settings['markersize']   = 3.0
        
    if(isinstance(histx_settings, type(None))):
        histx_settings = dict()
        histx_settings['xlim']      = [min(x), max(x)]
        histx_settings['weight']    = np.ones_like(x)
        histx_settings['color']     = 'blue'
        histx_settings['alpha']     = 0.5
        
    if(isinstance(histy_settings, type(None))):
        histy_settings = dict()
        histy_settings['ylim']      = [min(y), max(y)]
        histy_settings['weight']    = np.ones_like(y)
        histy_settings['color']     = 'green'
        histy_settings['alpha']     = 0.5
        
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, label=label, marker=ax_settings['marker'], c=ax_settings['color'], cmap=ax_settings['colormap'], alpha=ax_settings['alpha'], s=ax_settings['markersize'])
    #, facecolors=ax_settings['markerfacecolor'], edgecolors=ax_settings['markeredgecolor'], s=ax_settings['markersize'], alpha=ax_settings['alpha'])
    ax.set_xlim(histx_settings['xlim'])
    ax.set_ylim(histy_settings['ylim'])
    ax.legend()
    ax.grid('on', alpha=0.35, color='black', linestyle=':', linewidth=0.5)

    # now determine nice limits by hand:
    binwidth = np.abs( (histx_settings['xlim'][1] - histx_settings['xlim'][0] ) /101 )
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, weights=np.ones_like(x)*histx_settings['weight'], color=histx_settings['color'], lw=0, alpha=histx_settings['alpha']) # ,density=histx_settings['density']
    ax_histx.set_xlim(histx_settings['xlim'])
    ax_histy.hist(y, bins=bins, weights=np.ones_like(y)*histy_settings['weight'], color=histy_settings['color'], lw=0, alpha=histx_settings['alpha'], orientation='horizontal') # , density=histy_settings['density']
    ax_histy.set_ylim(histy_settings['ylim'])
    
    ax_histx.xaxis.grid('on', alpha=0.35, color='black', linestyle=':', linewidth=0.5)
    ax_histy.yaxis.grid('on', alpha=0.35, color='black', linestyle=':', linewidth=0.5)



# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

# plt.show()