import numpy as np

def linearKernel(x1, x2):
    """
    Returns a linear kernel between x1 and x2.
    Parameters
    ----------
    x1 : numpy ndarray
        A 1-D vector.
    x2 : numpy ndarray
        A 1-D vector of same size as x1.
    Returns
    -------
    : float
        The scalar amplitude.
    """
    return np.dot(x1, x2)