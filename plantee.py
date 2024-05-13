import numpy as np
import math

def calcNeumann(ex: np.array, ey: np.array, from_i, to_i, alpha, T_inf):
    """
    Computes the integral of the form-functions over a 3-node triangle element
        Me = int(s*N^T*N)dA

    Inputs:
        ex:     element x-coordinates
        ey:     element y-coordinates
        s:      constant scalar, e.g. density*thickness

    Outputs:
        Me:     integrated element matrix
    """
    if not ex.shape == (3,) or not ey.shape == (3,):
        raise Exception("Incorrect shape of ex or ey: {0}, {1} but should be (3,)".format(ex.shape, ey.shape))
    
    # Compute element area
    fbc = np.zeros((3, 1))
    fbc[from_i] = 1/2
    fbc[to_i] = 1/2
    L = math.sqrt((ex[to_i] - ex[from_i])**2 + (ey[to_i]-ey[from_i])**2)
    fbc*=-alpha*T_inf*L
    return fbc