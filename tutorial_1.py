# -*- coding: utf-8 -*-


import statistics_lib as st
import numpy as np



if __name__ == "__main__":
    mu = np.array([2,2])
    Sigma= np.array( [[1,0],[0,1]])
    data = np.array([2,1])
    print( st.multivariate_gaussian(data,mu,Sigma) )