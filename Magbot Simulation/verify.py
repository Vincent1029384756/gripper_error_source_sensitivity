import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import math
from simfuncs import motion_model_spring_damper
from LPfilter import LPfilter


numLinks = 2
linkLength = np.array([7.22e-3, 7.77e-3])
linkTwist = np.array([1.571, 0.0])
linkOffset = np.array([0.0, 0.0])
jointType = np.array([JointType.REV, JointType.REV])
magnetLocal = np.array([[0, 0, 0],
                        [36.859e-3, 0, 0], 
                        [-16.088e-3, 0, 0]])  #-16.088e-3
magnetPosLocal = np.array([[-3.2e-3, 0, 0],
                            [0.03585354, 0, 0], 
                            [-4.01e-3, 0, 0]]) #[-4.01e-3, 0.94e-3, 0] -3.8
T = np.array([[0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
K = np.array([[4e-4, 0],
                [0, 3.5e-4]])
c = np.array([[0.00098, 0],
                [0, 0.00069]])

#initialize msr
q = np.array([0.0,0.0])
msr = MSR(numLinks, linkLength, linkTwist, linkOffset, q, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

file_name = 'test1.csv'
#initialize csv file
row1 = np.array([0,0,0,0,0,0,0,0,0])
df = pd.DataFrame([row1])
df.to_csv(file_name, index = False, header = False)
u = np.array([1.89561600e-01, -5.70759024e+00, -2.17568997e+00, -5.93800982e+00, 5.07753668e+00, 4.94307821e-03, 4.81793281e+00, 2.25650804e+00])
#u = np.array([ 0.57437149, -2.6691442,  -1.37105932, -5.07627903 , 4.15286813, -0.44043183, 1.81565236 , 1.45294665])

t_total = 600
t_step = 0.1
for t in range(int(t_total/t_step)):
    #update torques and joint angles
    q, tau_u, tau_int, tau_s = motion_model_spring_damper(q, u, t_step, msr)

    q1 = q[0]*(180/math.pi)
    q2 = q[1]*(180/math.pi)


    #collects output
    output = np.array([(t)/10, q1, q2, tau_u[0], tau_u[1], tau_int[0], tau_int[1], tau_s[0], tau_s[1]])

        # Reshape to 2D array with one row
    output_reshaped = output.reshape(1, -1)

    # Append new output to CSV
    df = pd.DataFrame(output_reshaped)
    df.to_csv(file_name, mode='a', index = False, header = False)

df = pd.read_csv(file_name, header=None)

# plot joint angles over time
plt.figure(1)
plt.title('Joint Angles over Time')
plt.plot(df.iloc[:, 0], df.iloc[:, 1], label='Theta1')
plt.plot(df.iloc[:, 0], df.iloc[:, 2], label='Theta2')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint Angles (degrees)')
plt.legend()

# Plot torques
plt.figure(2)
plt.title('Torques vs time')
plt.subplot(2, 1, 1)
plt.title('Theta1')
plt.plot(df.iloc[:, 0], df.iloc[:, 3], label='tau_u')
plt.plot(df.iloc[:, 0], df.iloc[:, 5], label='tau_int')
plt.plot(df.iloc[:, 0], df.iloc[:, 7], label='tau_s')
plt.ylabel('Torque [Nm]')
plt.subplot(2, 1, 2)
plt.title('Theta2')
plt.plot(df.iloc[:, 0], df.iloc[:, 4], label='tau_u')
plt.plot(df.iloc[:, 0], df.iloc[:, 6], label='tau_int')
plt.plot(df.iloc[:, 0], df.iloc[:, 8], label='tau_s')

plt.xlabel('Time (seconds)')
plt.ylabel('Torque [Nm]')
plt.legend()

plt.show()