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

# Prediction plot
plt.scatter(data[:, 0], data[:, 1], c='black', marker="x", label="Input data")

y_gpy, sig_gpy, m_gpy = spacetime.models.gpytorch_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)
y_skl, sig_skl, m_skl = spacetime.models.sklearn_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)
y_pyk, sig_pyk, m_pyk = spacetime.models.pykrige_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)
y_gst, sig_gst, m_gst = spacetime.models.gstools_predict_1d(variogram_model=variogram_model, gridx=gridx, data=data)

plt.plot(gridx, y_gpy, c="tab:red", label="GPyTorch mean prediction")
plt.fill_between(gridx, y_gpy - 2*sig_gpy, y_gpy + 2*sig_gpy, edgecolor="tab:red", facecolor="None",linestyle="dashed", label="GPyTorch std prediction")
# plt.plot(gridx, c="tab:red", linestyle="dashed", label="GPyTorch std prediction")
plt.plot(gridx, y_skl, c="tab:blue", label="SciKit-Learn mean prediction")
plt.fill_between(gridx, y_skl+ 2*sig_skl, y_skl-2*sig_skl, edgecolor="tab:blue", facecolor="None",linestyle="dashed", label="SciKit-Learn std prediction")
plt.plot(gridx, y_pyk, c="tab:pink", label="PyKrige mean prediction")
plt.fill_between(gridx, y_pyk - 2*sig_pyk, y_pyk+2*sig_pyk, edgecolor="tab:pink", facecolor="None",linestyle="dashed", label="PyKrige std prediction")
plt.plot(gridx, y_gst, c="tab:brown", label="GSTools mean prediction")
plt.fill_between(gridx, y_gst-2*sig_gst, y_gst+2*sig_gst, edgecolor="tab:brown", facecolor="None", linestyle="dashed", label="GSTools std prediction")

plt.xlabel("1D coordinate")
plt.ylabel("Prediction")
plt.title("1D comparison: prediction")
plt.legend()
plt.ylim((-5, 5))
plt.show()

# Residuals plot

plt.figure()
y_gpy = np.array(y_gpy)
plt.plot(gridx, np.abs(y_skl - y_gpy), c="tab:blue", label="Abs. diff SciKit-Learn vs GPytorch")
plt.plot(gridx, np.abs(y_pyk - y_gpy), c="tab:pink", label="Abs. diff PyKrige vs GPyTorch")
plt.plot(gridx, np.abs(y_gst - y_gpy), c="tab:brown", label="Abs. diff GSTools vs GPyTorch")

plt.xlabel("1D coordinate")
plt.ylabel("Residuals")
plt.title("1D comparison: residuals")
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

yy_gpy, sig_gpy, mm_gpy = spacetime.models.gpytorch_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)
yy_pyk, sig_pyk, mm_pyk = spacetime.models.pykrige_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)
yy_skl, sig_skl, mm_skl = spacetime.models.sklearn_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)
yy_gst, sig_gst, mm_gst = spacetime.models.gstools_predict_2d(variogram_model=variogram_model, gridx=gridx, gridy=gridy, data=data)


kwargs_cmap_mean = {"vmin": 0, "vmax": 2, "cmap": "Spectral"}
import matplotlib
kwargs_cmap_res = {"cmap": "Reds", "norm": matplotlib.colors.LogNorm(vmin=10**(-8), vmax=0.1)}
plt.figure()
plt.title("2D comparison")
plt.axis('off')

plt.subplot(4, 2, 1)
plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], **kwargs_cmap_mean)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.colorbar()
plt.title("Input points")
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# def subplot_imshow(pos, pred, title, kwargs_cmap):
#
#     plt.subplot(*pos)
#     plt.imshow(pred, **kwargs_cmap, origin="lower")
#     plt.colorbar()
#     plt.title(title)
#     ax = plt.gca()
#     ax.axes.get_xaxis().set_ticks([])
#     ax.axes.get_yaxis().set_ticks([])
#
# subplot_imshow()

plt.subplot(4, 2, 2)
plt.imshow(yy_gpy.reshape((len(gridx), len(gridy))), **kwargs_cmap_mean, origin="lower")
plt.colorbar()
plt.title("GPyTorch")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 3)
plt.imshow(yy_pyk.reshape((len(gridx), len(gridy))), **kwargs_cmap_mean, origin="lower")
plt.colorbar()
plt.title("PyKrige")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 4)
yy_gpy = np.array(yy_gpy)
abs_diff_pyk = np.abs(yy_pyk.reshape((len(gridx), len(gridy))) - yy_gpy.reshape((len(gridx), len(gridy))))
plt.imshow(abs_diff_pyk, **kwargs_cmap_res, origin="lower")
plt.colorbar()
plt.title("Abs. diff PyKrige vs GPyTorch")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 5)
plt.imshow(yy_skl.reshape((len(gridx), len(gridy))), **kwargs_cmap_mean, origin="lower")
plt.colorbar()
plt.title("SciKit-Learn")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 6)
abs_diff_skl = np.abs(yy_skl.reshape((len(gridx), len(gridy))) - yy_gpy.reshape((len(gridx), len(gridy))))
plt.imshow(abs_diff_skl, **kwargs_cmap_res, origin="lower")
plt.colorbar()
plt.title("Abs. diff SciKit-Learn vs GPyTorch")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 7)
plt.imshow(yy_gst.reshape((len(gridx), len(gridy))), **kwargs_cmap_mean, origin="lower")
plt.colorbar()
plt.title("GSTools")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.subplot(4, 2, 8)
abs_diff_gst = np.abs(yy_gst.reshape((len(gridx), len(gridy))) - yy_gpy.reshape((len(gridx), len(gridy))))
plt.imshow(abs_diff_gst, **kwargs_cmap_res, origin="lower")
plt.colorbar()
plt.title("Abs. diff GSTools vs GPyTorch")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])