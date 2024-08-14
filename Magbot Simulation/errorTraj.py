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
print('*********************************************************************')

# name the csv file and specify the directory
#user_input_dir = '/mnt/newstorage/summer_project/results_dz1'
user_input_dir = '/home/vincent-gu/summer_project/results_dz1'
#user_input_file = input('Name the csv file: ')
user_input_file = 'result.csv'
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
headers = ['dx1', 'dy1', 'dz1', 'dx2', 'dy2', 'dz2', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]']
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


q_1 = np.array([0.0, 0.0])
q_2 = np.array([0.0, 0.0])

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
    input2 = input('Make your selection: ').lower()

    input3 = input('Are you sure about your choices?(y/n): ').lower()

    if input3 == 'y':
        sure = True
    elif input3 == 'n':
        sure = False

#retrieve coil current data
#path = '/mnt/newstorage/summer_project/Magbot Simulation'
path = '/home/vincent-gu/summer_project/Magbot Simulation'
data = pd.read_csv(os.path.join(path, 'coil_currents.csv'), header='infer')

# set translation based on user inputs
dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = 0

nf = int(0) #init figure number
for i in tqdm(np.arange(-5, 5.1, 0.1), desc="Process"):

    if input1 == 'a':
        if input2 == 'x':
            dx1 = i
            cx = 0
        elif input2 == 'y':
            dy1 = i
            cx = 1
        elif input2 == 'z':
            dz1 = i
            cx = 2
    
    elif input1 == 'b':
        if input2 == 'x':
            dx2 = i
            cx = 3
        elif input2 == 'y':
            dy2 = i
            cx = 4
        elif input2 == 'z':
            dz2 = i
            cx = 5
    
    #modify magLocal2
    magnetPosLocal_2 = np.array([[-3.2e-3, 0, 0],
                            [-3.72e-3 + dx1*1e-3, dy1*1e-3, dz1*1e-3], 
                            [-4.01e-3 + dx2*1e-3, dy2*1e-3, dz2*1e-3]])
    
    q_1 = np.array([0.0, 0.0])
    q_2 = np.array([0.0, 0.0])

    #initialize msr_2
    msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T)
    msr_2.m_change_magnets(magnetLocal, magnetPosLocal_2)
    msr_2.m_set_joint_stiffness(K, c)

    #actuate both msr and record trajectory
    q_init = np.radians(np.array([7, 0]))
    output1 = sf.simulate_magbot(data, q_init, msr_1)
    output2 = sf.simulate_magbot(data, q_init, msr_2)

    #calculate the errors
    dq1_max, dq1_mean, dq2_max, dq2_mean = sf.compare(output1, output2)

    output3 = np.array([dx1, dy1, dz1, dx2, dy2, dz2, dq1_max, dq1_mean, dq2_max, dq2_mean]).reshape(1, -1)
    df_to_append = pd.DataFrame(output3, \
                                columns=['dx1', 'dy1', 'dz1', 'dx2', 'dy2', 'dz2', \
                                         'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]'])
    df_to_append.to_csv(file_path, mode='a', index=False, header=False)

    # plot joint angles over time
    plt.figure(nf)
    plt.title('Joint Angles over Time')
    plt.plot(output1[:, 0], output1[:, 1], label='q1_base')
    plt.plot(output2[:, 0], output2[:, 1], label='q1_error')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    #directory = '/mnt/newstorage/summer_project/images_dz1'
    directory = '/home/vincent-gu/summer_project/images_dz1'
    filename1 = f'q1_dev{str(i)}.png'
    full_path1 = os.path.join(directory, filename1)
    plt.savefig(full_path1)
    plt.close()

    # plot joint angles over time
    plt.figure(nf+1)
    plt.title('Joint Angles over Time')
    plt.plot(output1[:, 0], output1[:, 2], label='q2_base')
    plt.plot(output2[:, 0], output2[:, 2], label='q2_error')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    filename2 = f'q2_dev{str(i)}.png'
    full_path2 = os.path.join(directory, filename2)
    plt.savefig(full_path2)
    plt.close()

    nf += 2

# plot error vs. std_dev
save_path = '/home/vincent-gu/summer_project/results/results.png'
#save_path = '/mnt/newstorage/summer_project/results/results.png'
plot_and_regress(file_path, cx, 6, 7, 8, 9, save_path)
print('*********************************************************************')