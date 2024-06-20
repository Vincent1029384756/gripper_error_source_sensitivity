import numpy as np
from math import cos, sin, radians

def t_ba(theta1, KP1_a, l1):
    alpha = (180-abs(theta1))/2
    delta_x = l1*cos(radians(alpha))
    delta_z = l1*sin(radians(alpha))

    if theta1 > 0:
        x = KP1_a[0] - delta_x
    else:
        x = KP1_a[0] + delta_x
    z = KP1_a[2] + delta_z
    y = KP1_a[1]
    return np.array([x, y, z])

def R_ba(theta1):
    theta1_rad = radians(theta1)
    return np.array([[cos(theta1_rad), 0, sin(theta1_rad)],
                     [0, 1, 0],
                     [-sin(theta1_rad), 0, cos(theta1_rad)]]).T


def t_cb(theta2, KP6_b, l2):
    R_db = np.array([[1, 0, 0],
                     [0, cos(radians(-10)), -sin(radians(-10))],
                     [0, sin(radians(-10)), cos(radians(-10))]])
    
    beta = theta2 + 10
    alpha = (180 - beta)/2
    delta_y = -l2*cos(radians(alpha))
    delta_z = l2*sin(radians(alpha))

    KP4_d = np.array([0, delta_y, delta_z])

    KP4_b = R_db@KP4_d + KP6_b

    return KP4_b

def R_cb(theta2):
    theta2 = radians(theta2)
    return np.array([[1, 0, 0],
                     [0, cos(theta2), -sin(theta2)],
                     [0, sin(theta2), cos(theta2)]])


def get_keypoints(theta1, theta2):
    # Arc lengths
    l1 = 3 #2.92370142
    l2 = 2.5 #2.43386663

    # Relative keypoint orientations (start data, this is true for every pose)
    KP0_a = np.array([0,0,0])
    KP1_a = np.array([0,0,4.65])
    KP2_b = np.array([0,0,0])
    KP3_b = np.array([0,0,14.1])#13.8
    KP4_c = np.array([0,0,0])
    KP5_c = np.array([0,0,7.2])  #6.82
    KP6_b = np.array([0,-0.26,4.79])

    # Perform calcs and check if it matches CAD data
    tba = t_ba(theta1, KP1_a, l1)
    Rba = R_ba(theta1)

    tcb = t_cb(theta2, KP6_b, l2)
    Rcb = R_cb(theta2)

    KP2_a = Rba@KP2_b + tba
    KP3_a = Rba@KP3_b + tba

    KP4_b = tcb
    KP5_b = Rcb@KP5_c + tcb
    KP4_a = Rba@KP4_b + tba
    KP5_a = Rba@KP5_b + tba

    keypoints_calc = np.concatenate((KP0_a, KP1_a, KP2_a, KP3_a, KP4_a, KP5_a), axis=0)
    keypoints_calc = np.reshape(keypoints_calc, (6, 3))
    return keypoints_calc

# Test keypoint locations, theta1 = 60deg, theta2 = 20 deg (from CAD model)

'''
KP0_gt = np.array([0,0,0])
KP1_gt = np.array([0,0,4.65])
KP2_gt = np.array([1,0,7.4])
KP3_gt = np.array([9.87,0,17.97])
KP4_gt = np.array([5.55,-1.09,12.82])
KP5_gt = np.array([8.37,-6.32,16.18])

keypoints_gt = np.concatenate((KP0_gt, KP1_gt, KP2_gt, KP3_gt, KP4_gt, KP5_gt), axis=0)
keypoints_gt = np.reshape(keypoints_gt, (6, 3))

# Joint angles 
theta1 = -40
theta2 = 50

keypoints_calc = get_keypoints(theta1, theta2)

print(keypoints_calc)
print(keypoints_gt - keypoints_calc)

'''