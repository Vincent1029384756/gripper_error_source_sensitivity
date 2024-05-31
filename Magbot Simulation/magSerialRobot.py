import numpy as np
from enum import Enum
from math import sin, cos, radians
import magfunctions as magF
from math import pi

class JointType(Enum):
     REV = 0
     PRIS = 1

class MagSerialRobot():
    def __init__(self, numLinks, linkLength, linkTwist, linkOffset, jointAngles, jointType, mTBase):
        self.jointAngles = jointAngles
        self.num_joints = len(self.jointAngles)
        self.num_links = numLinks
        self.linkLength = linkLength
        self.linkTwist = linkTwist
        self.linkOffset = linkOffset
        self.jointType = jointType
        self.mTBase = mTBase

        # Magnet data
        self.magnetLocal = None
        self.magnetPosLocal = None

        # Flex joint stiffness
        self.K_joints = None
        self.c_joints = None

        # Magnetization matrix
        self.Mcoil = np.array([[3.6,   -0.7,  -4.1, -17.2,   17.5,  -3.4,   1.7,   4.0],
                               [3.7,   18.1,   3.5,  -1.0,    0.8,  -4.0, -17.0,  -3.6],
                               [-0.7,   12.5,  -1.2,  12.2,   12.1,  -1.1,  12.1,  -1.1],
                               [-15.3,  153.5, -19.1, -79.5,  -93.5, -12.2, 154.8, -23.0],
                               [-38.3,    3.6,  41.2,  -6.3,   -0.8, -37.1,  15.7,  36.4],
                               [-8.3,   15.3,   9.9, 231.3, -227.4,   7.0, -11.5, -10.9],
                               [-18.7,  -90.1, -14.4, 149.2,  164.0, -20.3, -96.4, -13.9],
                               [-10.8, -247.7,  -9.4,   9.0,  -20.5,   9.7, 230.4,   8.7]])*(1e-3/24)
    
    def m_set_joint_angles(self, q):
        
        
        if q[0] < -radians(80):
            q[0] = -radians(80)
        if q[0] > radians(80):
            q[0] = radians(80)
        if q[1] < 0:
            q[1] = 0
        if q[1] > radians(80):
            q[1] = radians(80) 
        
        self.jointAngles = q

    def m_set_q1(self, q1):
        self.jointAngles[0] = q1
    
    def m_set_q2(self, q2):
        self.jointAngles[1] = q2

    def m_set_joint_stiffness(self, K, c):
        self.K_joints = K
        self.c_joints = c

    def m_calc_joint_force(self):
        '''
        Computes the generalized forces applied to joints due to the stiffness of the flex joints (TauS)
        '''
        return -self.K_joints@self.jointAngles

    def m_change_magnets(self, magnetLocal, magnetPosLocal):
        self.magnetLocal = magnetLocal
        self.magnetPosLocal = magnetPosLocal

    def m_set_coil_matrix(self, Mcoil):
        self.Mcoil = Mcoil

    def m_calc_transform_single_link(self, fromFrame):
        '''
        /* This function calculates the 4x4 homogeneous transformation matrix
        * from frame i to frame i-1.
        *
        * Inputs:
        * double linkLength - The length of link i (often symbolized by "a")
        *                     in units of length.
        * double linkTwist  - The twist of link i (often symbolized by
        *                     "alpha") in radians.
        * double linkOffset - The offset of the link i (often symbolized by
        *                     "d") in units of length.
        * double jointAnlge - The joint angle of link i (often symbolized by
        *                     "theta") in radians.
        * Outputs:
        * Eigen::Matrix4d T - A 4x4 homogeneous transformation matrix
        *                     (affine transformation of 3D points and
        *                     vectors) in mixed units.
        *
        * Details:
        * The transformation matrix is often written as T_i^j, where i is the
        * starting frame (the frame in which a point or vector is given) and
        * j is the ending frame (the frame in which we would like to express
        * the given point or vector). The most common transformation is from
        * a local link frame i to the base frame 0: T_i^0.
        * The Denavit-Hartenberg convention used here assumes that the ith
        * frame is rigidly attached to the ith link, with the z-axis collinear
        * with either the rotation axis (for a revolute joint) or the
        * translation axis (for a prismatic joint) of the joint connecting
        * links i and i+1.
        * The transformation matrix from link i to the preceding link i-1 can
        * be calculated from the Denavit-Hartenberg parameters of the ith
        * link. This transformation matrix is often written as A_i:
        *      T_i^{i-1} = A_i
        */
        '''
        
        theta = self.jointAngles[fromFrame] #rad
        alpha = self.linkTwist[fromFrame] #rad
        a = self.linkLength[fromFrame] #meter
        d = self.linkOffset[fromFrame] #meter
        
        T = np.array([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                      [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                      [0.0,             sin(alpha),             cos(alpha),            d],
                      [0.0,                    0.0,                    0.0,          1.0]])
        return T

    def m_calc_transform_mat(self, fromFrame, toFrame):
        '''
        /* This method returns the transformation matrix between two link
        * frames for the serial robot in its present state.
        *
        * Inputs:
        * int fromFrame       - The index of the frame in which coordinates
        *                       are presently expressed (local link frame).
        * int toFrame         - The index of the frame in which you want the
        *                       coordinates to be expressed. E.g. 0 is the
        *                       base frame.
        *
        * Outputs:
        * Eigen::Matrix4d T   - A 4x4 homogeneous transformation matrix
        *                       (affine transformation of 3D points and
        *                       vectors) in mixed units.
        *
        * Details:
        * The transformation matrix is often written as T_i^j, where i is the
        * starting frame (the frame in which a point or vector is given) and
        * j is the ending frame (the frame in which we would like to express
        * the given point or vector). The most common transformation is from
        * a local link frame i to the base frame 0: T_i^0.
        * The Denavit-Hartenberg convention used here assumes that the ith
        * frame is rigidly attached to the ith link, with the z-axis collinear
        * with either the rotation axis (for a revolute joint) or the
        * translation axis (for a prismatic joint) of the joint connecting
        * links i and i+1.
        * The transformation matrix from link i to the preceding link i-1 can
        * be calculated from the Denavit-Hartenberg parameters of the ith
        * link. This transformation matrix is often written as A_i:
        *      T_i^{i-1} = A_i
        * The transformation between two non-adjacent links (J<i) can then be
        * determined from the product of the A_i matrices:
        *      T_i^j = A_{j+1} * A_{j+2} ... * A_{i-1} * A_i
        * The transformation matrix has a rotation component R_i^j (the 3x3
        * rotation matrix) and a translation component o_i^j (the 3x1
        * translation vector):
        *      T_i^j = [R_i^j, o_i^j]
        *              [0,0,0,     1]
        * Conveniently, the inverse transformation matrix can be calculated
        * simply using the following formula:
        *      T_j^i = (T_i^j)^-1 = [(R_i^j)', -(R_i^j)'*o_i^j]
        *                           [   0,0,0,               1]
        * where ' denotes the matrix transpose (similar to MATLAB notation).
        */
        '''
        # The 4x4 homogeneous transformation matrix
        T = np.identity(4)
        # A 3x3 rotation matrix (used in finding the inverse of T)
        #Eigen::Matrix3d R;
        # A 3x1 displacement vector (used in finding the inverse of T)
        #Eigen::Vector3d o;
        # Assign the identity matrix to the transformation matrix (equivalent
        # to a null transformation). This is returned without modification if
        # fromFrame == toFrame.
        #T = Eigen::Matrix4d::Identity();
        # Check whether both of the requested frames are within the range from
        # 0 to numLinks
        if ((0 <= fromFrame) and (fromFrame <= self.num_links) and (0 <= toFrame) and (toFrame <= self.num_links)):
        
            # For now, assume that fromFrame - to Frame > 0
            # The starting frame is distal to the ending frame.
            for i in range(toFrame, fromFrame):
                T = T@self.m_calc_transform_single_link(i)
            
        else:        
            print("Requested frame exceeds number of links.")

        return T

    def m_calc_transform_mat_global(self, fromFrame):
        '''
        /* This method returns the transformation matrix from the specified
        * frame of the serial robot to the global frame.
        *
        * Inputs:
        * int fromFrame       - The index of the frame in which coordinates
        *                       are presently expressed (local link frame).
        *
        * Outputs:
        * Eigen::Matrix4d T   - A 4x4 homogeneous transformation matrix
        *                       (affine transformation of 3D points and
        *                       vectors) in mixed units.
        */
        '''
        return self.mTBase@self.m_calc_transform_mat(fromFrame, 0)

    def m_calc_unit_twist_global(self, jointNumber):
        '''
        /* This method calculates the twist of unit amplitude in global
        * coordinates corresponding to the joint specified by jointNumber in
        * the serial robot.
        *
        * Inputs:
        * int jointNumber - The number of the joint according to the DH
        *                   convention.
        *
        * Outputs:
        * Eigen::Matrix<double,6,1> jointTwist - The twist of unit magnitude
        *                                        corresponding to the joint.
        *
        * Details:
        * A twist is a screw $ = (w; vO) that describes the instantaneous
        * kinematics of a rigid body. I use the ray-coordinate convention to
        * describe screws. The 3x1 vector w is the angular velocity of the
        * rotating body. The 3x1 vector vO is the linear velocity of an
        * imaginary point on the body located at the origin of the present
        * frame:
        *      vO = v + r x w
        * where v is the linear velocity of the body at a point along the
        * rotational axis and r is a vector from the origin of the frame to
        * the rotational axis.
        * For a revolute joint with a reference frame defined with the DH
        * convention, the twist of unit magnitude corresponding to the ith
        * joint is
        *      $ = (0_z_(i-1); 0_o_(i-1) x 0_z_(i-1))
        * where 0_z_(i-1) is the z-axis unit vector of the i-1 frame
        * expressed in base frame (0 frame) coordinates, 0_o_(i-1) is the
        * position vector of the i-1 frame expressed relative to the base
        * frame, and "x" denotes the vector cross product.
        * Similarly, for a prismatic joint with a reference frame defined
        * with the DH convention, the twist of unit magnitude corresponding
        * to the ith joint is
        *      $ = (0, 0, 0; 0_z_(i-1))
        * For more details, see Davidson & Hunt, Robots and Screw Theory,
        * 2004.
        * Note: In the DH convention that I use, the ith joint precedes the
        * ith link. For example, the first link is the base link (Link 0),
        * which is connected to Link 1 by Joint 1. Link 1 then connects to
        * Link 2 by Joint 2.
        */
        '''
        
        frameNumber = jointNumber - 1
        TMat = self.m_calc_transform_mat_global(frameNumber)

        # We assume that joints are revolute
        angularVelocity = TMat[0:3, 2]
        position = TMat[0:3, 3]

        # Calculate the joint twist 
        jointTwist = np.concatenate((angularVelocity, np.cross(position, angularVelocity)), axis=0)

        return jointTwist

    def change_crew_order(self, screw):

        screwIntchngMat = np.array([[0,0,0,1,0,0],
                                    [0,0,0,0,1,0],
                                    [0,0,0,0,0,1],
                                    [1,0,0,0,0,0],
                                    [0,1,0,0,0,0],
                                    [0,0,1,0,0,0]])
        
        return screwIntchngMat@screw


    def m_calc_actuation_matrix(self):
        '''
        /* This function returns the magnetization matrix Mb that relates the
        * applied augmented magnetic field vector beta = [b;g] (8x1) to the
        * generalized forces Q at the joints of the robot
        *      Q = Mb * beta
        * with beta specified in global coordinates.
        *
        * Inputs:
        * None
        *
        * Outputs:
        * Eigen::Matrix<double,mNumLinks,8> Mb - nx8 actuation
        *                                                       matrix.
        * Details:
        * The instantaneous rate of work dW/dt of a wrench on a twist can be
        * determined using:
        *      dW/dt = $_t' * A * $_w
        * where $_t is a twist (w;vO) in column-vector form, ' denotes the
        * transpose, A is the screw interchange matrix, and $_w is a wrench
        * (f;tO) in column-vector form. The screw interchange matrix changes
        * the order of a screw from ray-coordinate order to coordinate-ray
        * order or vice-versa. As such, it doesn't matter what order the
        * wrench and twist are in, as long as they are both the same order.
        * For a serial robot, we can find the generalized force about a given
        * joint if we use the twist of unit amplitude coordesponding to that
        * joint:
        *      Q = $_t' * A * $_w
        * where Q is the generalized force in N.m or N for revolute or
        * prismatic joints, respectively.
        * For more details, see Davidson & Hunt, Robots and Screw Theory,
        * 2004.
        * For a robot with multiple wrenches acting in parallel (i.e.
        * multiple independent forces and torques acting on the robot in
        * different locations), all wrenches acting on links distal to the
        * joint of interest can be summed. This is convenient because
        * magnetic wrenches are linear functions of the applied augmented 8x1
        * field vector beta = [b;g]. As a result:
        *      $_{w,total,i} = sum_{j=i}^{n}($_w,j)
        *      $_{w,total,i} = sum_{j=i}^{n}(Mw_j) * beta
        * where $_{w,total,i} is the total wrench acting about Joint i and
        * Mw_j is the magnetic wrench matrix corresponding to Link j. From
        * this expression, a single linear relation can be derived that
        * relates the augmented magnetic field vector to the generalized
        * forces at the robot joints:
        *      Q_i = $_{t,i}' * A * sum_{j=i}^{n}(Mw_j) * beta
        *
        *      Mb = [$_{t,1}' * A * sum_{j=1}^{n}(Mw_j)]
        *           [$_{t,2}' * A * sum_{j=2}^{n}(Mw_j)]
        *                          :
        *                          :
        *           [$_{t,n}' * A * sum_{j=n}^{n}(Mw_j)]
        *
        *      Q = Mb * beta
        */
        '''

        Ma = np.zeros((self.num_links, 8))
        Mw = np.zeros((6, 8))
        #unitTwist = np.zeros(6)

        for i in range(self.num_links, 0, -1):
            
            unitTwist = self.m_calc_unit_twist_global(i) #Partially
            unitTwist = self.change_crew_order(unitTwist)
            Mw += magF.calc_dipole_wrench_matrix(self.m_get_magnet(i), self.m_get_magnet_pos(i))
            Ma[i-1, :] = np.transpose(unitTwist)@Mw

        return Ma@self.Mcoil
    
    def m_get_magnet(self, linkNumber):
        '''
        This function returns the dipole moment vector of the magnet in the
        * link specified in global coordinates.
        *
        * Inputs:
        * int linkNumber - The specified link (link 0 is the base link)
        *
        * Outputs:
        * Eigen::Vector3d m - 3x1 dipole moment vector in A.m^2
        '''
        T = self.m_calc_transform_mat_global(linkNumber)
        m = np.zeros(4)
        m[:3] = self.magnetLocal[linkNumber]
        m = T@m

        return m[:3]
    
    def m_get_magnet_pos(self, linkNumber):
        '''
        /* This function returns the position vector of the magnet in the
        * link specified in global coordinates.
        *
        * Inputs:
        * int linkNumber - The specified link (link 0 is the base link)
        *
        * Outputs:
        * Eigen::Vector3d r - 3x1 position vector in meters
        '''
        T = self.m_calc_transform_mat_global(linkNumber)
        r = np.ones(4)
        r[:3] = self.magnetPosLocal[linkNumber]
        r = T@r

        return r[:3]

    
    def m_calc_internal_gen_forces(self):
        '''
        /*
        * Calculate the generalized forces in the robot joints due to the
        * internal forces and torques between the magnets in the robot
        * links.
        *
        * Inputs:
        * None
        *
        * Outputs:
        * Eigen::Matrix<double, Eigen::Dynamic, 1> - nx1 vector of generalized
        *                                            forces in N and N.m
        *
        * Details:
        * The internal generalized force is the sum of wrenches from all
        * magnets proximal to a given joint on all magents distal to a
        * given joint. If i is the number of the generalized coordinate,
        * about which we wish to find the generalized force, then
        *
        * tau_{int,i} = twist_i * [DELTA] * sum_{j=i}^n (...
        *                      ...sum_{k=0}^{i-1} wrench_{k,j})
        *
        * Where twist_i is the unit twist corresponding to the ith generalized
        * coordinate, [DELTA] is the screw interchange matrix, and
        * wrench_{k,j} is the magnetic wrench caused by the field produced
        * by the magnet in link k on the magnet in link j.
        */
        '''

        tauInt = np.zeros(self.num_links)

        for linkNum in range(1, self.num_links+1):
            unitTwist = self.change_crew_order(self.m_calc_unit_twist_global(linkNum))

            for j in range(linkNum, self.num_links+1):
                mj = self.m_get_magnet(j)
                rj = self.m_get_magnet_pos(j)

                for k in range(linkNum):
                    mk = self.m_get_magnet(k)
                    rk = self.m_get_magnet_pos(k)
                    wrench = magF.calc_wrench_between_dipoles(mk, mj, rk, rj) 
                    tauInt[linkNum-1] += np.transpose(unitTwist)@wrench

        return tauInt

                

    

                

