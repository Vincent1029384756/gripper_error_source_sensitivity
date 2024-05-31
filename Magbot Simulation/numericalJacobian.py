import numpy as np
import sys
import copy


def evenSpacedOffsets(derivativeOrder, accuracyOrder, finiteDifferenceDirection=0):
    # TODO error checks for dOrder > 1, accuracyOrder > 1, dfDir e [0, -1, 1]
    numPts = int(2 * np.floor((derivativeOrder + 1) * 0.5) - 1 + accuracyOrder)
    if finiteDifferenceDirection == 0:
        increment = np.ceil(-numPts * 0.5)
    elif finiteDifferenceDirection == 1:
        numPts += 1
        increment = 0
    else: # fdDirection == -1
        numPts += 1
        increment = -(numPts + 1)
    offsets = np.zeros(numPts)
    for idx in range(numPts):
        offsets[idx] = idx + increment
    return offsets

def numDerCoeffs(derivativeOrder, offsets):
    # TODO error catch on derivative order, number of offsets
    scale = len(offsets)
    A = np.ones((scale, scale)) # row 0 is always ones
    A[1, :] = offsets # row 1 is always offsets
    for idx in range(2, scale):
        for jdx in range(scale):
            A[idx, jdx] = offsets[jdx] ** idx
    b = np.zeros(scale)
    b[derivativeOrder] = 1
    for idx in range(2, derivativeOrder + 1):
        b[derivativeOrder] *= idx
    try:
        coeffs = np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, ValueError):
        coeffs = np.linalg.lstsq(A, b, rcond=-1)[0]
    return coeffs

def numericalJacobian_offsets(x, fcn, offsets, stepMult=1e-6, *args, **kwargs):
    # TODO tests of fcn(x, *args, **kwargs) compatibility and return type
    weights = numDerCoeffs(1, offsets)
    sizeN = len(x)
    sizeM = len(fcn(x, *args, **kwargs))
    sizeW = len(weights)
    jacobian = np.zeros((sizeN, sizeM))
    for ndx in range(sizeN):
        step = max(abs(x[ndx] * stepMult), sys.float_info.epsilon)
        for wdx in range(sizeW):
            xPlus = copy.copy(x)
            xPlus[ndx] += offsets[wdx] * step
            jacobian[ndx, :] += weights[wdx] * fcn(xPlus, *args, **kwargs)
        jacobian[ndx, :] /= step
    return jacobian

def numericalJacobian(x, fcn, accuracyOrder=2, finiteDifferenceDirection=0, stepMult=1e-6, *args, **kwargs):
    return numericalJacobian_offsets(x,
                                     fcn,
                                     evenSpacedOffsets(1, accuracyOrder, finiteDifferenceDirection),
                                     stepMult,
                                     *args, **kwargs)


if __name__ == "__main__":
    def testFcn(x):
        return x * x * x
    x = np.array([1., 2., 3., 4., 5.])
    print(numericalJacobian(x, testFcn, accuracyOrder=2))
