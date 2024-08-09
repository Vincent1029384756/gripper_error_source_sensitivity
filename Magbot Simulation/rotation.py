import numpy as np
import math

def rot_x(theta_deg):

    #returns rotational matrix along the x-axis

    theta = math.radians(theta_deg)

    T = np.identity(4)
    T[2, 2] = math.cos(theta)
    T[1, 1] = math.cos(theta)
    T[1, 2] = math.sin(theta)
    T[2, 1] = -math.sin(theta)
    
    return T

def rot_y(theta_deg):

    #returns rotational matrix along the y-axis

    theta = math.radians(theta_deg)

    T = np.identity(4)
    T[0, 0] = math.cos(theta)
    T[2, 2] = math.cos(theta)
    T[0, 2] = -math.sin(theta)
    T[2, 0] = math.sin(theta)

    return T

def rot_z(theta_deg):

    #returns rotational matrix along the y-axis

    theta = math.radians(theta_deg)

    T = np.identity(4)
    T[0, 0] = math.cos(theta)
    T[1, 1] = math.cos(theta)
    T[0, 1] = -math.sin(theta)
    T[1, 0] = math.sin(theta)

    return T