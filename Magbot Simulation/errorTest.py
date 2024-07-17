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
print('***********************************************************************************')

#translations in mm
dx1 = float(input('dx1 = '))
dy1 = float(input('dy1 = '))
dz1 = float(input('dz1 = '))
dx2 = float(input('dx2 = '))
dy2 = float(input('dy2 = '))
dz2 = float(input('dz2 = '))

#let user name the csv file and specify the directory
user_input_dir = '/mnt/newstorage/summer_project/results'
user_input_file = input('Name the csv file: ')

#combine directory and file name to create the full path
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
headers = ['angle', 'delta_q1', 'delta_q2']
df = pd.DataFrame(columns=headers)
df.to_csv(file_path, index=False)

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

#define the error calculation function
def calc_error(msr_1, msr_2, qd, t_total):

    # generate current
    u = u_gen(qd, msr_1)

    t_step = 0.1

    #initialize q_1 and q_2
    q_1 = np.array([0.0,0.0])
    q_2 = np.array([0.0,0.0])

    for step in range(int(t_total/t_step)):
        #update torques and joint angles for msr_1
        q_1, _, _, _ = motion_model_spring_damper(q_1, u, t_step, msr_1)

        #update torques and joint angles for msr_2
        q_2, _, _, _ = motion_model_spring_damper(q_2, u, t_step, msr_2)

    delta_q = (q_2 - q_1)*(180/math.pi)

    return delta_q

#prompt for testing case
print("Which joint would u like to test")
print("a. joint 1 only")
print("b. joint 2 only")
print("c. both joints")
case = input("make your selection: ").lower()

# calculate the error at all angles between 0 and 90
for i in range(91):
    if case == 'a':
        qd_deg = np.array([i,0])
    
    elif case == 'b':
        qd_deg = np.array([0,i])
    
    elif case == 'c':
        qd_deg = np.array([i,i])
    
    else:
        print("Invalid selection.")
        break

    qd = np.radians(qd_deg)
    
    t_total = 120

    delta_q = calc_error(msr_1, msr_2, qd, t_total)

    #collect outputs
    delta_q1 = delta_q[0]
    delta_q2 = delta_q[1]
    output = np.array([i, delta_q1, delta_q2]).reshape(1, -1)
    df_to_append = pd.DataFrame(output, columns=['angle', 'delta_q1', 'delta_q2'])
    df_to_append.to_csv(file_path, mode='a', index=False, header=False)

    # Print delta_q for debugging
    #print(f"Angle {i}: delta_q = {delta_q}")

print(f"Data saved successfully to {file_path}")

print('***********************************************************************************')
