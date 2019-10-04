import matplotlib.pyplot as plt

def plotData(X, y, grid=False):
    """
    Plots the data points X and y into a new figure. Uses `+` for positive examples, and `o` for
    negative examples. `X` is assumed to be a Mx2 matrix
    Parameters
    ----------
    X : numpy ndarray
        X is assumed to be a Mx2 matrix.
    y : numpy ndarray
        The data labels.
    grid : bool (Optional)
        Specify whether or not to show the grid in the plot. It is False by default.
    Notes
    -----
    This was slightly modified such that it expects y=1 or y=0.
    """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    plt.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')
    plt.grid(grid)