import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import math
from simfuncs import motion_model_spring_damper
from LPfilter import LPfilter

'''
In this script, coil currents are gnerated based on open loop control.
The current OL controls works by generating an actuating torque on the gripper links,
such that the actuation torque is in equilibrium with the position dependent torque 
at the desired configuration:
tau_U = -tau_K
u can be calculated from the equation: M_u@u = tau_u
'''

'''
In our dynamics model, position dependent torque tau_K consists of two components:
tau_int: internal torque betweeen the two on board magnets
tau_S: torque due to joint stiffness
'''

#se up msr parameters
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

#Decide on desired joint angles (qd)
print('***********************************************************************************')
user_input1 = input("What is the desired first joint angel in degrees: ")
user_input2 = input("What is the desired second joint angel in degrees: ")

qd1 = float(user_input1)
qd2 = float(user_input2)
qd_degree = np.array([qd1, qd2])

#convert degrees to radians, as the rest of the calculations are carried out in radians
qd = np.radians(qd_degree)

#since we need to find tau_K at the desired configuration, we will use the desired angle
#when initializing msr
msr = MSR(numLinks, linkLength, linkTwist, linkOffset, qd, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

#calculate tau_int
tau_int = msr.m_calc_internal_gen_forces()
#calculate tau_S
tau_S = msr.m_calc_joint_force()

#calculate tau_U
tau_U = tau_int + tau_S

#now solve for u
M_u = msr.m_calc_actuation_matrix()
M_u_pinv = np.linalg.pinv(M_u)
u = -M_u_pinv@tau_U

'''
The output u is only a 8*1 matrix, when passing this output to the real coil, the magnetic field will
only last a very brief moment.
To provide a sustained current, we need to repeat u in the final output
'''

#let user nae the csv file
user_input3 = input("PLease enter a name for the csv file you want to save: ")
file_name = str(user_input3)

#wrriing the csv file
for i in range(200):
    
    #set up time
    index = i+1
    time = 67*i

    #adding index and time to each row
    new_number = np.array([index, time])
    row = np.concatenate((new_number, u)).reshape(1, -1)
    df = pd.DataFrame(row)
    
    #intial numpy array
    if i == 0:
        df.to_csv(file_name, index=False, header=False)
    
    #append other rows
    else:
        df.to_csv(file_name, mode='a', index=False, header=False)

print(f"Data saved successfully to {file_name}")
print('***********************************************************************************')