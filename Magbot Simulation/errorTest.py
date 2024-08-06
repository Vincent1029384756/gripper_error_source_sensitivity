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
from plot_and_regress import plot_and_regress
from tqdm import tqdm
import rotation as rot


#set up msr paremeters
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
T1 = np.array([[0, -1, 0, 0],
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
msr_1 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_1, jointType, T1)
msr_1.m_change_magnets(magnetLocal, magnetPosLocal)
msr_1.m_set_joint_stiffness(K, c)

print('*********************************************************************')
print('Define the current angle')
q1_deg = float(input('Q1: '))
q2_deg = float(input('Q2: '))

q1 = math.radians(q1_deg)
q2 = math.radians(q2_deg)

#collect user inputs
sure = False
while sure == False:
    print('along which axis do u wanna rotate the gripper \
          \na. x-axis \nb. y-axis \nc. z-axis')
    input1 = input('Make your selection: ')

    print('Which joint would you like to actuate: \na. Joint 1 \nb. Joint 2')
    input2 = input('Make your selection: ')

    check = input('Are you sure about your selections? (y/n)').lower()

    if check == 'y':
        sure = True
    
    else:
        sure = False

qd = np.array([q1, q2])
for a in tqdm(range(-3, 4, 1), desc="progress of script"):
    if input2 == 'a':
        qd[0] = a*math.radians(15)
    elif input2 == 'b' and a >= 0:
        qd[1] = a*math.radians(15)
    else:
        continue

    print(f'current qd: {np.degrees(qd)}')
    u = u_gen(qd, msr_1)
    print(u)
    
    #let user name the csv file and specify the directory
    user_input_dir = '/mnt/newstorage/summer_project/results'
    #user_input_dir = '/home/vincent-gu/summer_project/results'
    #user_input_file = input('Name the csv file: ')
    user_input_file = 'result.csv'
    file_path = os.path.join(user_input_dir, user_input_file)

    #initialize csv file
    headers = ['theta_x', 'theta_y', 'theta_z', 'delta_q1', 'delta_q2']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False)

    theta_x, theta_y, theta_z  = 0, 0, 0

    for i in tqdm(range(-20, 21), desc = 'current loop'):

        T2 = np.array([[0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

        theta = float(i)

        if input1 == 'a':
            theta_x = theta
            T2 = rot.rot_x(theta_x)@T2
            c_x = 0
        
        elif input1 == 'b':
            theta_y = theta
            T2 = rot.rot_y(theta_y)@T2
            c_x = 1
        
        elif input1 == 'c':
            theta_z = theta
            T2 = rot.rot_z(theta_z)@T2
            c_x = 2
        
        #initiate q1 and q2
        q_1 = np.array([q1, q2])
        q_2 = np.array([q1, q2])
        
        #initiate msr_2
        msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T2)
        msr_2.m_change_magnets(magnetLocal, magnetPosLocal)
        msr_2.m_set_joint_stiffness(K, c)

        #Pass u thru both msr and calculate the difference
        t_total = 500
        t_step = 0.1

        for t in range(int(t_total/t_step)):
            #update torques and joint angles for msr_1
            q_1, _, _, _ = motion_model_spring_damper(q_1, u, t_step, msr_1)

            #update torques and joint angles for msr_2
            q_2, _, _, _ = motion_model_spring_damper(q_2, u, t_step, msr_2)

        delta_q = (q_2 - q_1)*(180/math.pi)
        #print(np.degrees(q_2))
        #print(np.degrees(q_1))

        #record data
        output = np.array([theta_x, theta_y, theta_z, delta_q[0], delta_q[1]]).reshape(1, -1)
        df_to_append = pd.DataFrame(output, \
                                    columns=['theta_x', 'theta_y', 'theta_z', 'delta_q1', 'delta_q2'])
        df_to_append.to_csv(file_path, mode='a', index=False, header=False)
    
    print(f"Data saved successfully to {file_path}")
    print('close current figure to continue')

    plot_and_regress(file_path, c_x, 3, 4)

print('*********************************************************************')
