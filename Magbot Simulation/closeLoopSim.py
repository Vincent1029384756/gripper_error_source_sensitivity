# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:40:26 2024

@author: Vincent
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import math
from simfuncs import motion_model_spring_damper
from LPfilter import LPfilter

#desired joint angles
#We will take user input for this
user_input1 = input("What is your desired first joint angle in degrees: ")
user_input2 = input("What is your desired second joint angle in degrees: ")

#store inputs into float variables
q_d1 = float(user_input1)
q_d2 = float(user_input2)
#desired joint angles in degrees
q_dd = np.array([q_d1, q_d2])
#desired joint angles in radians
q_d = np.radians(q_dd)

#define initial joint angles
q = np.array([0.0,0.0])
qf = np.array([0.0,0.0])

#PID parameters:
Kp = 0.25e-3
Ki = 0.55e-3
Kd = 0.013e-3

#time step settings
dt = 0.1 #time step
total_time = 10
time = np.arange(0, total_time, dt)

#number of interations
iters = len(time)

#initialize output collection
output = np.zeros((iters,3))

#initialize the filtered output collection, this will be used to calculate derivative
qf = np.zeros((iters+1, 2))

#initialize the e_i term
integral = np.array([0.0,0.0])

# Define magnetic serial robot
numLinks = 2
linkLength = np.array([7.22e-3, 7.77e-3])
linkTwist = np.array([1.571, 0.0])
linkOffset = np.array([0.0, 0.0])
jointType = np.array([JointType.REV, JointType.REV])
magnetLocal = np.array([[0, 0, 0],
                        [35.859e-3, 0, 0], 
                        [-16.088e-3, 0, 0]])  #-16.088e-3
magnetPosLocal = np.array([[-3.2e-3, 0, 0],
                            [-3.72e-3, 0, 0], 
                            [-4.01e-3, 0, 0]]) #[-4.01e-3, 0.94e-3, 0] -3.8
T = np.array([[0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
K = np.array([[4e-4, 0],
                [0, 3.5e-4]])
c = np.array([[0.00098, 0],
                [0, 0.00069]])

msr = MSR(numLinks, linkLength, linkTwist, linkOffset, q, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

#Close loop control
for i in range(iters):
    #find the error term, as well the derivative and integral of the error term
    error = q_d - qf[i]
    integral += error * dt

    if i == 0:
        derivative = q_d/dt
    
    else:
        derivative = (qf[i]-qf[i-1]) / dt
    
    '''
    The desired torque can be calculated using the sum of -K_p*e, -K_i*e_1, and -K_d*e_dot
    '''
    tau_u = -Kp*error - Ki*integral - Kd*derivative

    '''
    The desired u for the current iteration can be solved from the equation:
    M_u@u = tau_u
    
    '''
    #before solving the actuation matrix, update the joint angles
    msr.m_set_joint_angles(qf[i])
    M_u = msr.m_calc_actuation_matrix()
    M_u_pinv = np.linalg.pinv(M_u)
    u = -M_u_pinv@tau_u
    '''
    Now that we have u calculated, we can calculate q with the function motion_model_spring_damper(q_prev, u, dt, msr)
    This function takes in the current joint angles q, u, as well as dt to return us the next q
    '''
    q_new, tau_u, tau_int, tau_s = motion_model_spring_damper(qf[i], u, dt, msr)

    #In real life, q_new will be noisy due to camera measurement errors
    #We will add normally distributed noise to q_new to simulate this effect
    std_dev = math.radians(5) #can be set to a different values
    noise = np.random.normal(q_new, std_dev, size=(1,2))
    
    #update q and add the noise
    q = q_new + noise

    #collect outputs
    output[i, 0] = time[i]
    output[i, 1:3] = q

    #calculate the low pass filter output
    qf[i+1] = LPfilter(i, output)
    #print(qf[i+1])

# %%
#Plot joint angle over time
plt.figure(1)
plt.title('Joint angle over time')
plt.plot(output[:, 0], np.degrees(output[:, 1]), label='Theta1')
plt.plot(output[:, 0], np.degrees(output[:, 2]), label='Theta2')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint Angles (degrees)')
plt.legend()
plt.show()