import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

%matplotlib inline
plt.rcParams["figure.figsize"] = (10,10)

k_1 = -0.04436
k_2 = -0.35894
k_3 =  0.14944

r = np.linspace(0,1,1000)
r_distorted = r*(1 + k_1 * r + k_2 * r**2 + k_3 * r**3)

plt.xlabel('Initial R')
plt.ylabel('Distorted R')
plt.plot(r,r_distorted)
plt.show()

def distort_line(x,y,k_1,k_2,k_3):
    r = np.sqrt(x**2 + y**2)
    x_distorted = x*(1 + k_1 * r + k_2 * r**2 + k_3 * r**3)
    y_distorted = y*(1 + k_1 * r + k_2 * r**2 + k_3 * r**3)
    return(x_distorted,y_distorted)

for y in np.linspace(-1,1,10):
    x = np.linspace(-1,1,1000)
    x_distorted,y_distorted = distort_line(x,y,k_1,k_2,k_3)
    plt.plot(x_distorted,y_distorted,color='k',alpha=0.8)
    
for x in np.linspace(-1,1,10):
    y = np.linspace(-1,1,1000)
    x_distorted,y_distorted = distort_line(x,y,k_1,k_2,k_3)
    plt.plot(x_distorted,y_distorted,color='k',alpha=0.8)
    
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

def undistort_point(undistortion_params,r_distorted):
    undistorted = r_distorted*(1 + undistortion_params[0] * r_distorted
                               + undistortion_params[1] * r_distorted**2
                               + undistortion_params[2] * r_distorted**3
                               + undistortion_params[3] * r_distorted**4
                               + undistortion_params[4] * r_distorted**5)
    return(undistorted)

def fun(undistortion_params,r_distorted):
    #Compute residuals.
    undistorted = undistort_point(undistortion_params, r_distorted)
    return((undistorted - np.linspace(0,1,1000))).ravel()
    
x0 = np.zeros(5).ravel()
res = least_squares(fun, x0,  verbose=2, ftol=1e-12,loss='linear', args=([r_distorted]))

undistorted = undistort_point(res.x,r_distorted)    
plt.plot(r_distorted,label='distorted',alpha=0.5)
plt.plot(undistorted,label='un distorted',alpha=0.5)
plt.plot(np.linspace(0,1,1000),label='target',alpha=0.5)
plt.legend()
plt.show()

print(res.x)
