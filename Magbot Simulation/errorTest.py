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
T = np.array([[0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
K = np.array([[4e-4, 0],
                [0, 3.5e-4]])
c = np.array([[0.00098, 0],
                [0, 0.00069]])

#retrieve coil current data
path = '/mnt/newstorage/summer_project/Magbot Simulation'
data = pd.read_csv(os.path.join(path, 'coil_currents.csv'), header='infer')

#initialize msr_1
q_init = np.array([0.0,0.0])
msr = MSR(numLinks, linkLength, linkTwist, linkOffset, q_init, jointType, T)
msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

#let user name the csv file and specify the directory
user_input_dir = '/mnt/newstorage/summer_project/results'
#user_input_dir = '/home/vincent-gu/summer_project/results'
#user_input_file = input('Name the csv file: ')
user_input_file = 'result.csv'
file_path = os.path.join(user_input_dir, user_input_file)

#initialize csv file
headers = ['std_dev[A]', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]']
df = pd.DataFrame(columns=headers)
df.to_csv(file_path, index=False)

print('*********************************************************************')

std_dev_range = float(input('What is the max deviation you want in the coil currents (Amp): '))
nf = int(0)

for std_dev in tqdm(np.arange(0, std_dev_range+1, 0.5), desc="progress"):

    q_init = np.radians(np.array([7, 0]))
    output1 = sf.simulate_magbot(data, q_init, msr)
    output2 = sf.simulate_error(data, q_init, msr, std_dev)

    dq1_max, dq1_mean, dq2_max, dq2_mean = sf.compare(output1, output2)

    output3 = np.array([std_dev, dq1_max, dq1_mean, dq2_max, dq2_mean]).reshape(1, -1)
    df_to_append = pd.DataFrame(output3, \
                                columns=['std_dev[A]', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]'])
    df_to_append.to_csv(file_path, mode='a', index=False, header=False)

    # plot joint angles over time
    plt.figure(nf)
    plt.title('Joint Angles over Time')
    plt.plot(output1[:, 0], output1[:, 1], label='q1_base')
    plt.plot(output2[:, 0], output2[:, 1], label='q1_error')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    directory = '/mnt/newstorage/summer_project/images'
    filename1 = f'q1_dev{str(std_dev)}.png'
    full_path1 = os.path.join(directory, filename1)
    plt.savefig(full_path1)

    # plot joint angles over time
    plt.figure(nf+1)
    plt.title('Joint Angles over Time')
    plt.plot(output1[:, 0], output1[:, 2], label='q2_base')
    plt.plot(output2[:, 0], output2[:, 2], label='q2_error')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    filename2 = f'q2_dev{str(std_dev)}.png'
    full_path2 = os.path.join(directory, filename2)
    plt.savefig(full_path2)

    nf += 2

# plot error vs. std_dev
plot_and_regress(file_path)
print('*********************************************************************')