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
from tqdm import tqdm
import simfuncs as sf
from plot_csv import plot_and_regress


'''
msr_1 is the baseline model
msr_2 is the modified model to simulate 
Divergence of the actual positions of the magnets from the designed layout

u_gen generates coil currents using msr_1, the basline
The same coil currents are passed through both msr_1 and msr_2
'''
print('***********************************************************************************')

# name the csv file and specify the directory
#user_input_dir = '/mnt/newstorage/summer_project/results'
user_input_dir = '/home/vincent-gu/summer_project/results'
#user_input_file = input('Name the csv file: ')
user_input_file = 'result.csv'
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
headers = ['std_dev[A]', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]']
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

#prompt user on which translation to make and in which direction
sure = False

while sure == False:
    print('Which magnet do you want to translate? \
        \na. Mag 1 \nb. Mag 2')
    input1 = input('Make your selection: ')

    print('In which direction(x/y/z)?')
    input2 = input('Make your selection(x/y/z): ').lower()

    input3 = input('Are you sure about your choices?(y/n): ').lower()

    if input3 == 'y':
        sure = True
    elif input3 == 'n':
        sure = False

