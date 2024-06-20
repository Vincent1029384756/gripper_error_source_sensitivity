import numpy as np
from simfuncs import motion_model_spring_damper
import pandas as pd
import math

'''
This function takes in q_d in radians, and returns a 8 by 1 coil curent matrix u
'''

def u_gen(qd, msr):
    #set joint angles to qd
    msr.m_set_joint_angles(qd)
    
    #calculate tau_int
    tau_int = msr.m_calc_tau_int()
    #calculate tau_S
    tau_S = msr.m_calc_joint_force()

    #calculate tau_U
    tau_U = -tau_int + tau_S

    #now solve for u
    M_u = msr.m_calc_actuation_matrix()
    M_u_pinv = np.linalg.pinv(M_u)
    u = -M_u_pinv@tau_U

    return u