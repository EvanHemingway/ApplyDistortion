import numpy as np
from scipy.optimize import least_squares

def findInverse(k):
    r = np.linspace(0,1,1000)
    r_distorted = r*(1 + k * r**2)
    
    x0 = np.zeros(5).ravel()
    res = least_squares(fun, x0,  verbose=0, ftol=1e-12,loss='linear', args=([r_distorted]))

    return res.x

def fun(undistortion_params,r_distorted):
    #Compute residuals.
    undistorted = undistort_point(undistortion_params, r_distorted)
    return((undistorted - np.linspace(0,1,1000))).ravel()

def undistort_point(undistortion_params,r_distorted):
    undistorted = r_distorted*(1 + undistortion_params[0] * r_distorted
                               + undistortion_params[1] * r_distorted**2
                               + undistortion_params[2] * r_distorted**3
                               + undistortion_params[3] * r_distorted**4
                               + undistortion_params[4] * r_distorted**5)
    return(undistorted)