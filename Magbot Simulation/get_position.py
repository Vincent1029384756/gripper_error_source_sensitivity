import numpy as np
from math import cos, sin, radians
from keypoint_model import R_cb, t_cb

inp = input("Theta_2: ")
theta2 = float(inp)

l2 = 2.5 #arc length of joint 2
KP6_b = np.array([0,-0.26,4.79]) #helper point

tcb = t_cb(theta2, KP6_b, l2)
Rcb = R_cb(theta2)

m2_c = np.array([0,0,2.5])
m2_b = Rcb@m2_c + tcb

print(m2_b)

pby = -2.5*cos(radians(-10))*cos(radians(85-(theta2/2)))-2.5*sin(radians(-10))*sin(radians(85-(theta2/2)))-0.26-2.5*sin(radians(theta2))
pbz = -2.5*sin(radians(-10))*cos(radians(85-(theta2/2)))+2.5*cos(radians(-10))*sin(radians(85-(theta2/2)))+4.79+2.5*cos(radians(theta2))
pb = np.array([0,pby,pbz])

print(pb)

tcb_c = t_cb(theta2, KP6_b, 0.5*l2) #centre of arc
print("Centre of Arc:")
print(tcb_c)

tcb_c_y = -1.25*cos(radians(-10))*cos(radians(85-(theta2/2)))-1.25*sin(radians(-10))*sin(radians(85-(theta2/2)))-0.26
tcb_c_z = -1.25*sin(radians(-10))*cos(radians(85-(theta2/2)))+1.25*cos(radians(-10))*sin(radians(85-(theta2/2)))+4.79
print(np.array([0,tcb_c_y, tcb_c_z]))