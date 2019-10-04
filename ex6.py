import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from svmTrain import svmTrain
from plotData import plotData
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualizeBoundary
from visualizeBoundaryLinear import visualizeBoundaryLinear
from dataset3Params import dataset3Params
from linearKernel import linearKernel

# =============== Part 1: Loading and Visualizing Data ================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and plot
# the data.
plt.ion()
print('Loading and Visualizing Data ...\n')
data = loadmat('ex6data1.mat')
X, y = data['X'], data['y'][:, 0]
plotData(X, y)
plt.show(block=False)
input('Program paused. Press enter to continue.')

# ==================== Part 2: Training Linear SVM ====================
# The following code will train a linear SVM on the dataset and plot the
# decision boundary learned.
print('Training Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
#clf = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=20)
#model = clf.fit(X, y.ravel())
model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
visualizeBoundaryLinear(X, y, model)
plt.show(block=False)
input('Program paused. Press enter to continue.')

# =============== Part 3: Implementing Gaussian Kernel ===============
# You will now implement the Gaussian kernel to use
# with the SVM. You should complete the code in gaussianKernel.py
print('Evaluating the Gaussian Kernel ...')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1],',
      'sigma = 0.5 : %f' % sim)
print('(this value should be about 0.324652)\n')
input('Program paused. Press enter to continue.')

# =============== Part 4: Visualizing Dataset 2 ================
# The following code will load the next dataset into your environment and
# plot the data.
print('Loading and Visualizing Data ...\n')
data = loadmat('ex6data2.mat')
X, y = data['X'], data['y'][:, 0]
plotData(X, y)
plt.show(block=False)
input('Program paused. Press enter to continue.')

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# After you have implemented the kernel, we can now use it to train the
# SVM classifier.
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

model= svmTrain(X, y, C, gaussianKernel, args=(sigma,))
visualizeBoundary(X, y, model)
plt.show(block=False)
input('Program paused. Press enter to continue.')

# =============== Part 6: Visualizing Dataset 3 ================
# The following code will load the next dataset into your environment and
# plot the data.
print('Loading and Visualizing Data ...\n')
data = loadmat('ex6data3.mat')
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]
plotData(X, y)
plt.show(block=False)
input('Program paused. Press enter to continue.')

# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
# This is a different dataset that you can use to experiment with. Try
# different values of C and sigma here.
C, sigma = dataset3Params(X, y, Xval, yval)
gamma = 1 / (2 * sigma**2)
model = svmTrain(X, y, C, gaussianKernel, args=(sigma,))
visualizeBoundary(X, y, model)
plt.show(block=False)
input('Program paused. Press enter to continue.')