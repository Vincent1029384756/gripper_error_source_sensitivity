import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR


#set up msr parameters
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

q = np.array([0,0])

#initialize msr
msr = MSR(numLinks, linkLength, linkTwist, linkOffset, q, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

q_deg = np.array([0.0,0.0])

for i in range(91):

    q = np.radians(q_deg)
    msr.m_set_joint_angles(q)

    #calculate tau_int
    tau_int = msr.m_calc_internal_gen_forces()

    if i == 0:
        #initialize the CSV file in which we will save the data
        file_name = 'tau_int_dipole.csv'
        output = np.array([q_deg[1], tau_int[1]])
        df = pd.DataFrame([output])
        df.to_csv(file_name, index = False, header = False)
    else:
        output = np.array([q_deg[1], tau_int[1]])
        df = pd.DataFrame([output])
        df.to_csv(file_name, mode='a', index = False, header = False)
    #update joint angle
    q_deg += 1

#plot output
df = pd.read_csv(file_name, header=None)

#plot tau_int vs. theta2
df = pd.read_csv(file_name, header=None)
plt.figure(1)
plt.title('tau_int vs. theta2')
plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=None)
plt.xlabel('Theta2 [deg]')
plt.ylabel('tau-int [N*m]')

plt.show()
