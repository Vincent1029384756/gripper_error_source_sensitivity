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
import rotation as rt

'''
msr_1 is the baseline model
msr_2 is the modified model to simulate 
Divergence of the actual positions of the magnets from the designed layout

u_gen generates coil currents using msr_1, the basline
The same coil currents are passed through both msr_1 and msr_2
'''
print('*********************************************************************')

base_root = '/mnt/newstorage/summer_project/'
usr_input1 = input('Name the folder where you want to save the final results: ')
usr_input2 = input('Name the folder where you wanna save all the trajectory images: ')
results_path = os.path.join(base_root, usr_input1)
images_path = os.path.join(base_root, usr_input2)

# Create the directories
os.makedirs(results_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

user_input_file = 'result.csv'
file_path = os.path.join(results_path, user_input_file)

#initialize csv file
headers = ['theta1x','theta1y','theta1z','theta2x','theta2y','theta2z', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]']
df = pd.DataFrame(columns=headers)
df.to_csv(file_path, index=False)

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


q_1 = np.array([0.0, 0.0])
q_2 = np.array([0.0, 0.0])

#initialize msr_1
msr_1 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_1, jointType, T)
msr_1.m_change_magnets(magnetLocal1, magnetPosLocal)
msr_1.m_set_joint_stiffness(K, c)

sure = False
while sure == False:
    
    print('Select which angle of mag1 u like to change \
        \n0. theta1_x \n1. theta1_y \n2. theta1_z \n3. None')

    input1 = input("Make your selctions: ")

    print('Select which angle of mag2 u like to change \
        \n0. theta2_x \n1. theta2_y \n2. theta2_z \n3. None')

    input2 = input("Make your selctions: ")

    check = input('Are you sure about your selections? (y/n)').lower()

    if check == 'y':
        sure = True
    
    else:
        sure = False

indices1 = input1.replace(',', ' ').split()
indices1 = [int(index) for index in indices1]

indices2 = input2.replace(',', ' ').split()
indices2 = [int(index) for index in indices2]

#retrieve coil current data
path = '/mnt/newstorage/summer_project/Magbot Simulation'
data = pd.read_csv(os.path.join(path, 'coil_currents.csv'), header='infer')

#initializing rotational variables
theta1_deg = np.array([0, 0, 0])
theta2_deg = np.array([0, 0, 0])

#initialize magLocal2
magnetLocal2 = np.zeros((3,3))

#figure number
nf = int(0)

for i in tqdm(range(-20, 20), desc="current loop"):

    #update rotations
    if indices1 != [3]:
        theta1_deg[int(input1)] = i
        cx = int(input1)
    
    elif indices2 != [3]:
        theta2_deg[int(input2)] = i
        cx = int(input2) + 3
    

    magnetLocal2_1 = rt.rot_z(theta1_deg[2])@rt.rot_y(theta1_deg[1]) @rt.rot_x(theta1_deg[0])\
          @ np.array([35.859e-3, 0, 0, 1])
    magnetLocal2_2 = rt.rot_z(theta2_deg[2])@rt.rot_y(theta2_deg[1]) @rt.rot_x(theta2_deg[0])\
          @ np.array([-16.088e-3, 0, 0, 1])
    
    magnetLocal2[1, :] = magnetLocal2_1[: 3]
    magnetLocal2[2, :] = magnetLocal2_2[: 3]

    #initialize msr2
    msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T)
    msr_2.m_change_magnets(magnetLocal2, magnetPosLocal)
    msr_2.m_set_joint_stiffness(K, c)

    #actuate both msr and record trajectory
    q_init = np.radians(np.array([7, 0]))
    output1 = sf.simulate_magbot(data, q_init, msr_1)
    output2 = sf.simulate_magbot(data, q_init, msr_2)

    #calculate the errors
    dq1_max, dq1_mean, dq2_max, dq2_mean = sf.compare(output1, output2)
    output3 = np.concatenate((theta1_deg, theta2_deg\
                              , np.array([dq1_max, dq1_mean, dq2_max, dq2_mean]))).reshape(1, -1)

    df_to_append = pd.DataFrame(output3, \
                                columns=['theta1x','theta1y','theta1z','theta2x','theta2y','theta2z', \
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
    filename1 = f'q1_dev{str(i)}.png'
    full_path1 = os.path.join(images_path, filename1)
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
    full_path2 = os.path.join(images_path, filename2)
    plt.savefig(full_path2)
    plt.close()

    nf += 2

# plot error vs. std_dev
#save_path = '/home/vincent-gu/summer_project/results/results.png'
save_path = os.path.join(results_path, 'results.png')
plot_and_regress(file_path, cx, 6, 7, 8, 9, save_path)
print('*********************************************************************')