import numpy as np
from math import pi


mu0 = pi * 4e-7

def calc_xprod_skew_mat(v):
    '''
    /*
     * This function returns the skew-symmetric matrix of a vector v that
     * is equivalent to the cross product.
     *
     * Inputs:
     * Eigen::Vector3d v - 3x1 vector
     *
     * Outputs:
     * Eigen::Matrix3d M - 3x3 matrix
     *
     * Details:
     * The cross product between two 3x1 vectors is usually represented as:
     *      u = v x w
     * However, it can be convenient to express the cross product as a
     * matrix multiplication:
     *      u = M(v)w
     * Where M is a 3x3 skew-symmetric matrix:
     *      M = [  0 -v3  v2]
     *          [ v3   0 -v1]
     *          [-v2  v1   0]
     * See p. 59 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
     */
    '''
    M = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return M

def calc_dipole_torque_matrix(dipoleMoment):
    '''
    /*
     * This function returns the matrix Mt that defines the torque
     * experienced by a dipole moment in a magnetic field.
     *
     * Inputs:
     * Eigen::Vector3d dipoleMoment - 3x1 point dipole moment vector in A.m^2
     *
     * Outputs:
     * Eigen::Matrix3d Mt - 3x3 dipole torque matrix in N.m/T
     *
     * Details:
     * See p. 68 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
     */
    '''
    return calc_xprod_skew_mat(dipoleMoment)

def calc_dipole_force_matrix(dipoleMoment):
    '''
    /* This function returns the matrix Mf that defines the force
     * experienced by a dipole moment in a magnetic gradient.
     *
     * Inputs:
     * Eigen::Vector3d dipoleMoment - 3x1 point dipole moment vector in A.m^2
     *
     * Outputs:
     * Eigen::Matrix3d Mf - 3x5 dipole torque matrix in N.m/T
     *
     * Details:
     * At a point in space the gradient of the magnetic field can be fully
     * defined by only five gradients. By convention, the following
     * gradients are often used:
     *      g = [g_xx
     *           g_xy
     *           g_xz
     *           g_yy
     *           g_yz]
     * Where g_uv = dbu/dv (d being the partial derivative operator).
     * The relationship between the force on a magnetic dipole and the
     * magnetic field gradient is linear:
     *      f = Mf g
     * This function returns Mf given the dipole moment vector.
     * See p. 68 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
     */
    '''
    return np.array([[dipoleMoment[0], dipoleMoment[1], dipoleMoment[2],                0,               0],
                     [             0, dipoleMoment[0],               0,  dipoleMoment[1], dipoleMoment[2]],
                     [-dipoleMoment[2],               0, dipoleMoment[0], -dipoleMoment[2], dipoleMoment[1]]])


def calc_dipole_wrench_matrix(dipoleMoment, position):
    '''
    /* This function returns the matrix Mw that defines the wrench
     * experienced by a dipole moment in a magnetic field and gradient.
     *
     * Inputs:
     * Eigen::Vector3d dipoleMoment - 3x1 point dipole moment vector in A.m^2
     * Eigen::Vector3d position     - 3x1 position vector of the point
     *                                dipole in m.
     *
     * Outputs:
     * Eigen::Matrix<double,6,8> Mw - 6x8 dipole wrench matrix in mixed
     *                                units (N.m/T, N.m^2/T, N/T)
     *
     * Details:
     * A magnetic point dipole m located at a position r experiences both
     * a force and a torque in response to the field gradient and field
     * vector, respectively, at the position r. The force and torque can
     * be combined into a single 6x1 entity called a wrench, which has
     * mixed units (N and N.m).
     * A wrench is a screw $ = (f;tO) that describes the instantaneous
     * dynamics of a rigid body. I use the ray-coordinate convention to
     * describe screws. The 3x1 vector f is the force applied to the body.
     * The 3x1 vector tO is the torque applied about an imaginary point on
     * the body located at the origin of the present frame:
     *      tO = t + r x f
     * where t is the torque on the body at a point along the line of
     * action of the force and r is a vector from the origin of the frame
     * to the line of action.
     * For more details, see Davidson & Hunt, Robots and Screw Theory,
     * 2004.
     * Using wrench notation allows a linear expression to be derived that
     * expresses the wrench applied to a magnetic dipole exposed to an
     * 8x1 augmented field vector beta = [b;g]:
     *      $ = Mw * beta
     */
    '''

    Mt = calc_dipole_torque_matrix(dipoleMoment)
    Mf = calc_dipole_force_matrix(dipoleMoment)
    Mw = np.zeros((6, 8))
    Mw[:3, 3:8] = Mf
    Mw[3:, :3] = Mt
    Mw[3:, 3:8] = calc_xprod_skew_mat(position)@Mf

    return Mw    

def calc_dipole_field(dipoleMoment, relPosition):
    '''
    /* This function returns the magnetic field vector at a point in space
     * defined by a relative position vector and the dipole moment
     * vector.
     *
     * Inputs:
     * Eigen::Vector3d dipoleMoment - 3x1 point dipole moment vector in A.m^2
     * Eigen::Vector3d position     - 3x1 position vector in m
     *
     * Outputs:
     * Eigen::Vector3d field        - 3x1 magnetic field vector in tesla
     *
     * Details:
     * See p. 60 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
    '''
    distance = np.linalg.norm(relPosition)
    field = mu0/(4*pi*pow(distance,5.0))*(3*np.expand_dims(relPosition, axis=1)@np.expand_dims(relPosition, axis=0) - pow(distance,2.0)*np.identity(3))@dipoleMoment
    return field


def calc_force_between_dipoles(m1Vec, m2Vec, r12Vec):
    '''
    * Returns the force on dipole 2 due to dipole 1.
     *
     * Inputs:
     * Eigen::Vector3d m1Vec  - 3x1 point dipole vector 1 in A.m^2
     * Eigen::Vector3d m2Vec  - 3x1 point dipole vector 2 in A.m^2
     * Eigen::Vector3d r12Vec - 3x1 position of dipole 2 relative to dipole 1
     *                          (i.e. P2 - P1) in m
     *
     * Outputs:
     * Eigen::Vector3d force  - 3x1 force vector on point dipole 2 in N
     *
     * Details:
     * See p. 68 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
    '''
    r12 = np.linalg.norm(r12Vec)
    
    r12Hat = r12Vec / r12
    force = 3 * mu0 / (4*pi*pow(r12,4.0))* (np.dot(r12Hat, m2Vec)*m1Vec + r12Hat.dot(m1Vec)*m2Vec \
                + (np.dot(m1Vec, m2Vec) - 5.0*(np.dot(r12Hat, m1Vec)*np.dot(r12Hat, m2Vec)))*r12Hat)
    
    return force

def calc_torque_between_dipoles(m1Vec, m2Vec, r12Vec):
    '''
    * Returns the torque on dipole 2 due to dipole 1.
     *
     * Inputs:
     * Eigen::Vector3d m1Vec  - 3x1 point dipole vector 1 in A.m^2
     * Eigen::Vector3d m2Vec  - 3x1 point dipole vector 2 in A.m^2
     * Eigen::Vector3d r12Vec - 3x1 position of dipole 2 relative to dipole 1
     *                          (i.e. P2 - P1) in m
     *
     * Outputs:
     * Eigen::Vector3d torque  - 3x1 force vector on point dipole 2 in N
     *
     * Details:
     * See p. 69 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
     * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
    '''

    b1 = calc_dipole_field(m1Vec, r12Vec)
    return np.cross(m2Vec, b1)

    


def calc_wrench_between_dipoles(m1Vec, m2Vec, r1Pos, r2Pos):
    '''
    * Returns the wrench on dipole 2 due to dipole 1.
    *
    * Inputs:
    * Eigen::Vector3d m1Vec   - 3x1 point dipole vector 1 in A.m^2
    * Eigen::Vector3d m2Vec   - 3x1 point dipole vector 2 in A.m^2
    * Eigen::Vector3d r1Pos   - 3x1 position of dipole 1 in m
    * Eigen::Vector3d r2Pos   - 3x1 position of dipole 2 in m
    *
    * Outputs:
    * Eigen::VectorXd wrench  - 6x1 force vector on point dipole 2 in ray-
    *                           coordinate order (f;tO) in N and N.m
    *
    * Details:
    * See p. 69 Abbott et al., "Magnetic Methods in Robotics," Ann. Rev.
    * Ctrl. Robot. Auton. Syst., vol. 3, pp. 57-90, 2020.
    '''

    r12Vec = r2Pos - r1Pos
    force12 = calc_force_between_dipoles(m1Vec, m2Vec, r12Vec) # N # Produces NaN!!!
    torque12 = calc_torque_between_dipoles(m1Vec, m2Vec,r12Vec) # N.m #Produces Nan!!!
    term2 = torque12 + np.cross(r2Pos, force12)
    wrench = np.concatenate((force12, term2), axis=0) # (N; N.m)
    
    return wrench