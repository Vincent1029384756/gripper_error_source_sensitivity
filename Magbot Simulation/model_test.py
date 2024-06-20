import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import math
from simfuncs import motion_model_spring_damper
from LPfilter import LPfilter
from current_gen import u_gen
import time

'''
In this script, coil currents are gnerated based on open loop control.
The current OL controls works by generating an actuating torque on the gripper links,
such that the actuation torque is in equilibrium with the position dependent torque 
at the desired configuration:
tau_U = -tau_K
u can be calculated from the equation: M_u@u = tau_u
'''

'''
In our dynamics model, position dependent torque tau_K consists of two components:
tau_int: internal torque betweeen the two on board magnets
tau_S: torque due to joint stiffness
'''

#se up msr parameters
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

#Decide on desired joint angles (qd)
print('***********************************************************************************')


#initialize msr
q = np.array([0.0,0.0])
msr = MSR(numLinks, linkLength, linkTwist, linkOffset, q, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

#let user name the csv file
user_input = input("PLease enter a name for the csv file you want to save: ")
file_name = str(user_input)

#initialize csv file
row1 = np.array([0,0,0,0,0,0,0,0,0])
df = pd.DataFrame([row1])
df.to_csv(file_name, index = False, header = False)

cont = True
i = 1

while cont == True:
    user_input1 = input("What is the desired first joint angel in degrees: ")
    user_input2 = input("What is the desired second joint angel in degrees: ")
    user_input3 = input("How long you want to actuate the gripper in this current setting (in seconds): ")

    qd1 = float(user_input1)
    qd2 = float(user_input2)
    qd_degree = np.array([qd1, qd2])

    #convert degrees to radians, as the rest of the calculations are carried out in radians
    qd = np.radians(qd_degree)

    t_total = float(user_input3)

    #geberate u
    u = u_gen(qd, msr)

    #show user the current coil current setting
    print("Current coil current setting")
    print(u)

    #actuate msr
    #compute torques and configurations of each time step, then record them on the dedicated csv file
    
    t_step = 0.1

    for t in range(int(t_total/t_step)):
        #update torques and joint angles
        q, tau_u, tau_int, tau_s = motion_model_spring_damper(q, u, t_step, msr)

        q1 = q[0]*(180/math.pi)
        q2 = q[1]*(180/math.pi)


        #collects output
        output = np.array([(t+i)/10, q1, q2, tau_u[0], tau_u[1], tau_int[0], tau_int[1], tau_s[0], tau_s[1]])

         # Reshape to 2D array with one row
        output_reshaped = output.reshape(1, -1)

        # Append new output to CSV
        df = pd.DataFrame(output_reshaped)
        df.to_csv(file_name, mode='a', index = False, header = False)

    #determine if user wants to continue
    user_input4 = input("Do you want to continue (y/n): ")
    input4 = str(user_input4)
    if input4 == 'y' or input4 == 'Y':
        cont = True
    elif input4 == 'n' or input4 == 'N':
        cont = False
    else:
        print('Invalid input')
    
    i += t_total/t_step


print(f"Data saved successfully to {file_name}")
print('***********************************************************************************')

#plot output
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