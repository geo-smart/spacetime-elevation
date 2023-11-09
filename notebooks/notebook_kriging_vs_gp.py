"""Comparing prediction between kriging and GP"""

import numpy as np
import matplotlib.pyplot as plt

import spacetime.models
from spacetime import models

# 1D comparison

# Data from https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/05_kriging_1D.html and
# https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
data = np.array([
     [-5.01, 1.06], [-4.90, 0.92], [-4.82, 0.35], [-4.69, 0.49], [-4.56, 0.52],
     [-4.52, 0.12], [-4.39, 0.47], [-4.32,-0.19], [-4.19, 0.08], [-4.11,-0.19],
     [-4.00,-0.03], [-3.89,-0.03], [-3.78,-0.05], [-3.67, 0.10], [-3.59, 0.44],
     [-3.50, 0.66], [-3.39,-0.12], [-3.28, 0.45], [-3.20, 0.14], [-3.07,-0.28],
     [-3.01,-0.46], [-2.90,-0.32], [-2.77,-1.58], [-2.69,-1.44], [-2.60,-1.51],
     [-2.49,-1.50], [-2.41,-2.04], [-2.28,-1.57], [-2.19,-1.25], [-2.10,-1.50],
     [-2.00,-1.42], [-1.91,-1.10], [-1.80,-0.58], [-1.67,-1.08], [-1.61,-0.79],
     [-1.50,-1.00], [-1.37,-0.04], [-1.30,-0.54], [-1.19,-0.15], [-1.06,-0.18],
     [-0.98,-0.25], [-0.87,-1.20], [-0.78,-0.49], [-0.68,-0.83], [-0.57,-0.15],
     [-0.50, 0.00], [-0.38,-1.10], [-0.29,-0.32], [-0.18,-0.60], [-0.09,-0.49],
     [0.03 ,-0.50], [0.09 ,-0.02], [0.20 ,-0.47], [0.31 ,-0.11], [0.41 ,-0.28],
     [0.53 , 0.40], [0.61 , 0.11], [0.70 , 0.32], [0.94 , 0.42], [1.02 , 0.57],
     [1.13 , 0.82], [1.24 , 1.18], [1.30 , 0.86], [1.43 , 1.11], [1.50 , 0.74],
     [1.63 , 0.75], [1.74 , 1.15], [1.80 , 0.76], [1.93 , 0.68], [2.03 , 0.03],
     [2.12 , 0.31], [2.23 ,-0.14], [2.31 ,-0.88], [2.40 ,-1.25], [2.50 ,-1.62],
     [2.63 ,-1.37], [2.72 ,-0.99], [2.80 ,-1.92], [2.83 ,-1.94], [2.91 ,-1.32],
     [3.00 ,-1.69], [3.13 ,-1.84], [3.21 ,-2.05], [3.30 ,-1.69], [3.41 ,-0.53],
     [3.52 ,-0.55], [3.63 ,-0.92], [3.72 ,-0.76], [3.80 ,-0.41], [3.91 , 0.12],
     [4.04 , 0.25], [4.13 , 0.16], [4.24 , 0.26], [4.32 , 0.62], [4.44 , 1.69],
     [4.52 , 1.11], [4.65 , 0.36], [4.74 , 0.79], [4.84 , 0.87], [4.93 , 1.01],
     [5.02 , 0.55]
])

gridx = np.linspace(-6, 6, 200)

variogram_model = {"model_name": "gaussian", "range": 2, "sill": 2, "nugget": 0.0001}

plt.scatter(data[:, 0], data[:, 1], c='black', marker="x", label="Input data")

y_gpytorch, m_gpytorch = spacetime.models.gpytorch_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)
y_sklearn, m_sklearn = spacetime.models.sklearn_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)
y_pykrige, m_pykrige = spacetime.models.pykrige_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)

plt.plot(gridx, y_gpytorch, c="tab:red", label="GPyTorch prediction")
plt.plot(gridx, y_sklearn, c="tab:blue", label="SciKit-Learn prediction")
plt.plot(gridx, y_pykrige, c="tab:pink", label="PyKrige prediction")

plt.xlabel("1D coordinate")
plt.ylabel("Prediction")
plt.title("1D comparison: Kriging vs GP")
plt.legend()
plt.ylim((-5, 5))
plt.show()

# 2D comparison

data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ]
)

gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)

yy_gpytorch, mm_gpytorch = spacetime.models.gpytorch_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)
yy_pykrige, mm_pykrige = spacetime.models.pykrige_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)
yy_sklearn, mm_sklearn = spacetime.models.sklearn_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)


kwargs_cmap = {"vmin": 0, "vmax": 2, "cmap": "Spectral"}
plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], **kwargs_cmap)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.colorbar()
plt.title("Input points")

plt.subplot(2, 2, 2)
plt.imshow(yy_gpytorch.reshape((len(gridx), len(gridy))), **kwargs_cmap, origin="lower")
plt.colorbar()
plt.title("GPyTorch")

plt.subplot(2, 2, 3)
plt.imshow(yy_pykrige.reshape((len(gridx), len(gridy))), **kwargs_cmap, origin="lower")
plt.colorbar()
plt.title("PyKrige")

plt.subplot(2, 2, 4)
plt.imshow(yy_sklearn.reshape((len(gridx), len(gridy))), **kwargs_cmap, origin="lower")
plt.colorbar()
plt.title("SciKit-Learn")

