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
headers = ['Tx[m]', 'Ty[m]', 'Tz[m]', 'dq1_max[deg]', 'dq1_mean[deg]', 'dq2_max[deg]', 'dq2_mean[deg]']
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


q_1 = np.array([0.0, 0.0])
q_2 = np.array([0.0, 0.0])

#initialize msr_1
msr_1 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_1, jointType, T1)
msr_1.m_change_magnets(magnetLocal, magnetPosLocal)
msr_1.m_set_joint_stiffness(K, c)

sure = False
while sure == False:
    
    print('Select in which direction u wanna move the gripper \na. x \nb. y \
          \nc. z')
    input1 = input('Make your selection: ').lower()

    check = input('Are you sure about your selections? (y/n)').lower()

    if check == 'y':
        sure = True
    
    else:
        sure = False

#retrieve coil current data
data = pd.read_csv(os.path.join(base_root, 'Magbot Simulation/coil_currents.csv'), header='infer')

Tx, Ty, Tz = 0, 0, 0

#figure number
nf = int(0)
value = 0

for i in tqdm(range(-10, 11), desc='current loop'):

    T2 = np.array([[0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if input1 == 'a':
        Tx = i*1e-2
        T2[0, 3] += Tx
        c_x = 0
        value = Tx
    
    elif input1 == 'b':
        Ty = i*1e-2
        T2[1, 3] += Ty
        c_x = 1
        value = Ty

    elif input1 == 'c':
        Tz = i*1e-2
        T2[2, 3] += Tz
        c_x = 2
        value = Tz

    #initiate msr_2
    msr_2 = MSR(numLinks, linkLength, linkTwist, linkOffset, q_2, jointType, T2)
    msr_2.m_change_magnets(magnetLocal, magnetPosLocal)
    msr_2.m_set_joint_stiffness(K, c)

    #actuate both msr and record trajectory
    q_init = np.radians(np.array([7, 0]))
    output1 = sf.simulate_magbot(data, q_init, msr_1)
    output2 = sf.simulate_magbot(data, q_init, msr_2)

    #calculate the errors
    dq1_max, dq1_mean, dq2_max, dq2_mean = sf.compare(output1, output2)
    output3 = np.array([Tx, Ty, Tz, dq1_max, dq1_mean, dq2_max, dq2_mean]).reshape(1, -1)

    df_to_append = pd.DataFrame(output3, \
                                columns=['Tx[m]', 'Ty[m]', 'Tz[m]',
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
    filename1 = f'q1_dev{str(value)}.png'
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
    filename2 = f'q2_dev{str(value)}.png'
    full_path2 = os.path.join(images_path, filename2)
    plt.savefig(full_path2)
    plt.close()

    nf += 2

# plot error vs. std_dev
#save_path = '/home/vincent-gu/summer_project/results/results.png'
save_path = os.path.join(results_path, 'results.png')
plot_and_regress(file_path, c_x, 3, 4, 5, 6, save_path)
print('*********************************************************************')