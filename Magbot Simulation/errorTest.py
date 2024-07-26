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

print('#####################################################################')
print('Define the current angle')
q1_deg = float(input('Q1: '))
q2_deg = float(input('Q2: '))

q1 = math.radians(q1_deg)
q2 = math.radians(q2_deg)

#collect user inputs
sure = False
while sure == False:
    print('Select which magnetization to change: \na. Mag1 \nb. Mag2')
    input1 = input('Make your selection: ')

    print('Which joint would you like to actuate: \na. Joint 1 \nb. Joint 2')
    input2 = input('Make your selection: ')

    check = input('Are you sure about your selections? (y/n)').lower()

    if check == 'y':
        sure = True
    
    else:
        sure = False

qd = np.array([q1, q2])
for i in tqdm(range(-3, 4, 1), desc="progress of script"):
    if input2 == 'a':
        qd[0] = i*math.radians(15)
    elif input2 == 'b':
        qd[1] = i*math.radians(15)
    
    print(f'current qd: {np.degrees(qd)}')
    u = u_gen(qd, msr_1)
    print(u)

    magnetLocal2 = np.array([[0, 0, 0],
                        [35.859e-3, 0, 0], 
                        [-16.088e-3, 0, 0]])
    
    #let user name the csv file and specify the directory
    user_input_dir = '/mnt/newstorage/summer_project/results'
    #user_input_file = input('Name the csv file: ')
    user_input_file = 'result.csv'
    file_path = os.path.join(user_input_dir, user_input_file)

    #initialize csv file
    headers = ['mag1_diver', 'mag2_diver', 'delta_q1', 'delta_q2']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False)