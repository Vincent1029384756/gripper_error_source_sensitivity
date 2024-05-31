import numpy as np
from magSerialRobot import JointType
from magSerialRobot import MagSerialRobot as MSR
import pandas as pd
import os
import matplotlib.pyplot as plt
from simfuncs import simulate_magbot
from math import radians as rad

# Define magnetic serial robot
numLinks = 2
linkLength = np.array([7.22e-3, 7.77e-3])
linkTwist = np.array([1.571, 0.0])
linkOffset = np.array([0.0, 0.0])
jointAngle = np.array([0.0, 0.0])
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

msr = MSR(numLinks, linkLength, linkTwist, linkOffset, jointAngle, jointType, T)

msr.m_change_magnets(magnetLocal, magnetPosLocal)
msr.m_set_joint_stiffness(K, c)

# Import coil current data
path = '/home/wen-gu/vincent_model/Magbot Simulation' # Change this to what it is on your computer
data = pd.read_csv(os.path.join(path, 'test2.csv'), header='infer')
 
# Save estimated stuff
output = simulate_magbot(data, np.array([rad(7), 0]), msr)


# plot joint angles over time
plt.figure(1)
plt.title('Joint Angles over Time')
plt.plot(output[:, 0], output[:, 1], label='Theta1')
plt.plot(output[:, 0], output[:, 2], label='Theta2')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint Angles (degrees)')
plt.legend()


# Plot torques
plt.figure(2)
plt.title('Torques vs time')
plt.subplot(2, 1, 1)
plt.title('Theta1')
plt.plot(output[:, 0], output[:, 3], label='tau_u')
plt.plot(output[:, 0], output[:, 5], label='tau_int')
plt.plot(output[:, 0], output[:, 7], label='tau_s')
plt.ylabel('Torque [Nm]')
plt.subplot(2, 1, 2)
plt.title('Theta2')
plt.plot(output[:, 0], output[:, 4], label='tau_u')
plt.plot(output[:, 0], output[:, 6], label='tau_int')
plt.plot(output[:, 0], output[:, 8], label='tau_s')

plt.xlabel('Time (seconds)')
plt.ylabel('Torque [Nm]')
plt.legend()

plt.show()
