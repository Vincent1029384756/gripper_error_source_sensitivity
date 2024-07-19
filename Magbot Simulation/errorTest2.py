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

#set up msr paremeters
numLinks = 2
linkLength = np.array([7.22e-3, 7.77e-3])
linkTwist = np.array([1.571, 0.0])
linkOffset = np.array([0.0, 0.0])
jointType = np.array([JointType.REV, JointType.REV])
magnetLocal1 = np.array([[0, 0, 0],
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

#initialize msr_1
q_1 = np.array([0.0,0.0])
q_2 = np.array([0.0,0.0])
msr_1 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_1, jointType, T)
msr_1.m_change_magnets(magnetLocal1, magnetPosLocal)
msr_1.m_set_joint_stiffness(K, c)

qd = np.array([1.57079633, 1.57079633])

#u = u_gen(qd, msr_1)
#print(u)
u = np.array([-1.78768686, -8.68299844, -1.43886932,  0.42682159, -1.11292936,  1.83812089, 7.77649816, 1.66477169])


# Prompt user for mag1 and mag2 rotations

theta1_deg = np.array([0, 0, 0])
theta2_deg = np.array([0, 0, 0])

theta1 = np.radians(theta1_deg)
theta2 = np.radians(theta2_deg)

print('#####################################################################')

sure = False
while sure == False:

    print('Select which angle(s) of mag1 u like to change \nseperate by spaces or commas \
        \n0. theta1_x \n1. theta1_y \n2. theta1_z \n3. None')

    input1 = input("Make your selctions: ")

    print('Select which angle(s) of mag2 u like to change \nseperate by spaces or commas \
        \n0. theta2_x \n1. theta2_y \n2. theta2_z \n3. None')

    input2 = input("Make your selctions: ")

    check = input('Are you sure about your selections? (y/n)').lower()

    if check == 'y':
        sure = True
    
    else:
        sure = False

#let user name the csv file and specify the directory
user_input_dir = '/mnt/newstorage/summer_project/results'
#user_input_file = input('Name the csv file: ')
user_input_file = 'result.csv'
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
headers = ['theta1x','theta1y','theta1z','theta2x','theta2y','theta2z', 'delta_q1', 'delta_q2']
df = pd.DataFrame(columns=headers)
df.to_csv(file_path, index=False)

print("Hang in tight buddy, working on it")

indices1 = input1.replace(',', ' ').split()
indices1 = [int(index) for index in indices1]

indices2 = input2.replace(',', ' ').split()
indices2 = [int(index) for index in indices2]

for i in range(30):

    #update rotations
    if indices1 != [3]:
        for a in indices1:
            theta1[a] += math.radians(1)
            print(theta1)
    
    if indices2 != [3]:
        for b in indices2:
            theta2[b] += math.radians(1)
            print(theta2)

    q_1 = np.array([0.0,0.0])
    q_2 = np.array([0.0,0.0])

    magnet1x = 35.859e-3*math.cos(theta1[1])*math.cos(theta1[2])
    magnet1y = 35.859e-3*math.cos(theta1[1])*math.sin(theta1[2])
    magnet1z = 35.859e-3*math.sin(theta1[1])

    magnet2x = -16.088e-3*math.cos(theta2[1])*math.cos(theta2[2])
    magnet2y = -16.088e-3*math.cos(theta2[1])*math.sin(theta2[2])
    magnet2z = -16.088e-3*math.sin(theta2[1])

    magnetLocal2 = np.array([[0, 0, 0],
                            [magnet1x, magnet1y, magnet1z], 
                            [magnet2x, magnet2y, magnet2z]])  #-16.088e-3
    
    print(magnetLocal2)
    
    #initializ msr_2
    msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T)
    msr_2.m_change_magnets(magnetLocal2, magnetPosLocal)
    msr_2.m_set_joint_stiffness(K, c)

    #Pass u thru both msr and calculate the difference
    t_total = 120
    t_step = 0.1

    for t in range(int(t_total/t_step)):
        #update torques and joint angles for msr_1
        q_1, _, _, _ = motion_model_spring_damper(q_1, u, t_step, msr_1)

        #update torques and joint angles for msr_2
        q_2, _, _, _ = motion_model_spring_damper(q_2, u, t_step, msr_2)

    delta_q = (q_2 - q_1)*(180/math.pi)
    print(np.degrees(q_2))
    print(np.degrees(q_1))

    #record current delta_q
    output = np.concatenate((np.degrees(theta1), np.degrees(theta2), delta_q)).reshape(1, -1)
    df_to_append = pd.DataFrame(output, columns=['theta1x','theta1y','theta1z','theta2x','theta2y','theta2z', 'delta_q1', 'delta_q2'])
    df_to_append.to_csv(file_path, mode='a', index=False, header=False)

print(f"Data saved successfully to {file_path}")

print('#####################################################################')