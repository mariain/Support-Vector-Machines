import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

def visualizeBoundaryLinear(X, y, model):
    """
    Plots a linear decision boundary learned by the SVM.
    Parameters
    ----------
    X : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).
    y : array_like
        (m, ) The data labels.
    model : dict
        Dictionary of model variables learned by SVM.
    """
    w, b = model['w'], model['b']
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0] * xp + b)/w[1]

    plotData(X, y)
    plt.plot(xp, yp, '-b')