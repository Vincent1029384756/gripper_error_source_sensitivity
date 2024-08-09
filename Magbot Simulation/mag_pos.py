import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import math
from simfuncs import motion_model_spring_damper
import os
from current_gen import u_gen

'''
msr_1 is the baseline model
msr_2 is the modified model to simulate 
Divergence of the actual positions of the magnets from the designed layout

u_gen generates coil currents using msr_1, the basline
The same coil currents are passed through both msr_1 and msr_2
'''

#translations in mm
dx1 = 0
dx2 = 5
dy1 = 0
dy2 = 0
dz1 = 0
dz2 = 0

#set up msr paremeters
numLinks = 2
linkLength = np.array([7.22e-3, 7.77e-3])
linkTwist = np.array([1.571, 0.0])
linkOffset = np.array([0.0, 0.0])
jointType = np.array([JointType.REV, JointType.REV])
magnetLocal = np.array([[0, 0, 0],
                        [35.859e-3, 0, 0], 
                        [-16.088e-3, 0, 0]])  #-16.088e-3
magnetPosLocal_1 = np.array([[-3.2e-3, 0, 0],
                            [-3.72e-3, 0, 0], 
                            [-4.01e-3, 0, 0]]) #[-4.01e-3, 0.94e-3, 0] -3.8
magnetPosLocal_2 = np.array([[-3.2e-3, 0, 0],
                            [-3.72e-3 + dx1*1e-3, dy1*1e-3, dz1*1e-3], 
                            [-4.01e-3 + dx2*1e-3, dy2*1e-3, dz2*1e-3]]) #[-4.01e-3, 0.94e-3, 0] -3.8
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

q_1 = np.array([0.0,0.0])
q_2 = np.array([0.0,0.0])

#initialize msr_1
msr_1 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_1, jointType, T)
msr_1.m_change_magnets(magnetLocal, magnetPosLocal_1)
msr_1.m_set_joint_stiffness(K, c)

#initialize msr_2
msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T)
msr_2.m_change_magnets(magnetLocal, magnetPosLocal_2)
msr_2.m_set_joint_stiffness(K, c)

#let user name the csv file and specify the directory
user_input_dir = '/mnt/newstorage/summer_project/results'
user_input_file = 'result.csv'

#combine directory and file name to create the full path
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
row1 = np.array([0,0,0,0,0])
df = pd.DataFrame([row1])
df.to_csv(file_path, index=False, header=False)

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
    u = u_gen(qd, msr_1)

    #show user the current coil current setting
    print("Current coil current setting")
    print(u)

    #actuate msr
    #compute torques and configurations of each time step, then record them on the dedicated csv file
    
    t_step = 0.1

    for t in range(int(t_total/t_step)):
        #update torques and joint angles for msr_1
        q_1, _, _, _ = motion_model_spring_damper(q_1, u, t_step, msr_1)

        q1_1 = q_1[0]*(180/math.pi)
        q2_1 = q_1[1]*(180/math.pi)

        #update torques and joint angles for msr_2
        q_2, _, _, _ = motion_model_spring_damper(q_2, u, t_step, msr_2)

        q1_2 = q_2[0]*(180/math.pi)
        q2_2 = q_2[1]*(180/math.pi)

        #collects output
        output = np.array([(t+i)/10, q1_1, q2_1, q1_2, q2_2])

         # Reshape to 2D array with one row
        output_reshaped = output.reshape(1, -1)

        # Append new output to CSV
        df = pd.DataFrame(output_reshaped)
        df.to_csv(file_path, mode='a', index = False, header = False)
    
    delta_q = (q_2 - q_1)*(180/math.pi)

    print(f"Delta_q = {delta_q}")

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


#print(f"Data saved successfully to {file_path}")

print('***********************************************************************************')

#plot output
df = pd.read_csv(file_path, header=None)

# plot joint 1 angles over time
plt.figure(1)
plt.title('Joint 1 Angle Comparison')
plt.plot(df.iloc[:, 0], df.iloc[:, 1], label='Theta1_original')
plt.plot(df.iloc[:, 0], df.iloc[:, 3], label='Theta1_modified')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint 1 Angle (degrees)')
plt.legend()

# plot joint 2 angles over time
plt.figure(2)
plt.title('Joint 2 Angle Comparison')
plt.plot(df.iloc[:, 0], df.iloc[:, 2], label='Theta2_original')
plt.plot(df.iloc[:, 0], df.iloc[:, 4], label='Theta2_modified')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint 2 Angle (degrees)')
plt.legend()

#plt.show()