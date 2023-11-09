"""
Functions to run kriging or Gaussian Process models in 1D time, 2D space and 3D space+time
"""

import numpy as np

import gpytorch
import torch
import sklearn.gaussian_process
from pykrige.ok import OrdinaryKriging

from spacetime import conversions

# 1/ GPyTorch models and prediction fixing the covariance parameters
# Correspondance with kriging:! a GPyTorch kernel is always a ScaleKernel (variogram sill) * BaseKernel (variogram form)

def gpytorch_1d_model(train_x, train_y, base_kernel, lengthscale, outputscale):

    # Exact GP model in 1D (see https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)
    class Exact1DModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact1DModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Exact1DModel(train_x, train_y, likelihood)

    # Fix covariance parameters
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.0001),
        'mean_module.constant': torch.mean(train_y),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(outputscale),
    }
    model.initialize(**hypers)

    return model

def gpytorch_2d_shared_model(train_x, train_y, base_kernel, lengthscale, outputscale):

    # Shared 2D kernel
    class Exact2DShared(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact2DShared, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel(active_dims=(0,1)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Exact2DShared(train_x, train_y, likelihood)

    # Fix covariance parameters
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.001),
        'mean_module.constant': torch.mean(train_y),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(outputscale),
    }
    model.initialize(**hypers)

    return model

def gpytorch_2d_additive_model(train_x, train_y, base_kernel, lengthscale, outputscale):

    # Additive 2D kernel
    class Exact2DAdditive(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact2DAdditive, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel(active_dims=(0,))) + \
                                gpytorch.kernels.ScaleKernel(base_kernel(active_dims=(1,)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Exact2DAdditive(train_x, train_y, likelihood)

    # Initialize parameters for additive kernel
    model.likelihood.noise_covar.noise = torch.tensor(0.0001)
    model.mean_module.constant = torch.mean(train_y)
    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[0].outputscale = torch.tensor(outputscale)
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[1].outputscale = torch.tensor(outputscale)

    return model

def gpytorch_2d_product_model(train_x, train_y, base_kernel, lengthscale, outputscale):

    # Product 2D kernel
    class Exact2DProduct(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact2DProduct, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel(active_dims=(0,))) * \
                                gpytorch.kernels.ScaleKernel(base_kernel(active_dims=(1,)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Exact2DProduct(train_x, train_y, likelihood)

    # Initialize parameters for product kernel
    model.likelihood.noise_covar.noise = torch.tensor(0.0001)
    model.mean_module.constant = torch.mean(train_y)
    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[0].outputscale = torch.tensor(np.sqrt(outputscale))
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[1].outputscale = torch.tensor(np.sqrt(outputscale))

    return model


def gpytorch_predict_1d(variogram_model, gridx, data):
    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Convert x/y grid into
    grid_coords = torch.from_numpy(gridx)

    # X data is single coordinates (1 column)
    train_x = torch.from_numpy(data[:, 0])
    # Y data are values to predict (1 column)
    train_y = torch.from_numpy(data[:, 1])

    base_kernel = getattr(gpytorch.kernels, base_kernel_name + "Kernel")  # GPyTorch adds "Kernel" to each name

    # Define model and fix covariance parameters
    model = gpytorch_1d_model(train_x=train_x, train_y=train_y, base_kernel=base_kernel,
                              lengthscale=lengthscale, outputscale=outputscale)

    # Predict
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.eval()  # We go in eval mode without training
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(grid_coords))  # And predict on the grid coordinates

    means = observed_pred.mean

    return means, model


def gpytorch_predict_2d(variogram_model, gridx, gridy, data):

    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Convert x/y grid into
    x, y = np.meshgrid(gridx, gridy)
    grid_coords = torch.from_numpy(np.dstack((x.flatten(), y.flatten())).squeeze())

    # X data are coordinates (2 columns)
    train_x = torch.from_numpy(data[:, 0:2])
    # Y data are values to predict (1 column)
    train_y = torch.from_numpy(data[:, 2])

    base_kernel = getattr(gpytorch.kernels, base_kernel_name + "Kernel")  # GPyTorch adds "Kernel" to each name
    # Define model and fix covariance parameters
    model = gpytorch_2d_shared_model(train_x=train_x, train_y=train_y, base_kernel=base_kernel,
                                       lengthscale=lengthscale, outputscale=outputscale)

    # Predict
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.eval()  # We go in eval mode without training
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(grid_coords))  # And predict on the grid coordinates

    means = observed_pred.mean

    return means, model

# 2/ SciKit-Learn GP model and prediction fixing the covariance

def sklearn_predict_1d(variogram_model, gridx, data):

    # Convert parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Define kernel
    kernel = sklearn.gaussian_process.kernels.ConstantKernel(outputscale) * \
             getattr(sklearn.gaussian_process.kernels, base_kernel_name)(lengthscale)

    # Define regressor, without optimizer to fix the parameters
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=0.0001)

    # Transform y to be centered on the mean manually
    y_in = data[:, 1]
    mean_y = np.mean(y_in)
    y_in = y_in - mean_y

    gpr.fit(data[:, 0].reshape(-1, 1), y_in.reshape(-1, 1))

    y_pred, sigma = gpr.predict(gridx.reshape(-1, 1), return_std=True)
    y_pred = np.squeeze(y_pred)

    # Transform back
    y_pred += mean_y

    return y_pred, gpr


def sklearn_predict_2d(variogram_model, gridx, gridy, data):

    # Convert parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Define kernel
    kernel = sklearn.gaussian_process.kernels.ConstantKernel(outputscale) * \
             getattr(sklearn.gaussian_process.kernels, base_kernel_name)([lengthscale, lengthscale])

    # Define regressor, without optimizer to fix the parameters
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=0.0001)

    # Transform y to be centered on the mean manually
    y_in = data[:, 2]
    mean_y = np.mean(y_in)
    y_in = y_in - mean_y

    gpr.fit(data[:, 0:2], y_in.reshape(-1, 1))

    x, y = np.meshgrid(gridx, gridy)
    grid_coords = torch.from_numpy(np.dstack((x.flatten(), y.flatten())).squeeze())

    y_pred, sigma = gpr.predict(grid_coords, return_std=True)
    y_pred = np.squeeze(y_pred)

    # Transform back
    y_pred += mean_y

    return y_pred, gpr


# 3/ PyKrige models and prediction fixing the covariance

def pykrige_predict_1d(variogram_model, gridx, data):

    model_name = variogram_model["model_name"]
    variogram_parameters = {k: variogram_model[k] for k in ["sill", "range", "nugget"]}

    # Define model forcing variogram parameters
    OK = OrdinaryKriging(
        data[:, 0],
        np.zeros(data.shape[0]),
        data[:, 1],
        variogram_model=model_name,
        variogram_parameters=variogram_parameters,
        verbose=False,
        enable_plotting=False,
    )

    # Predict on grid
    z, ss = OK.execute("grid", gridx, np.array([0.0]))
    z = np.squeeze(z)

    return z, OK


def pykrige_predict_2d(variogram_model, gridx, gridy, data):

    model_name = variogram_model["model_name"]
    variogram_parameters = {k: variogram_model[k] for k in ["sill", "range", "nugget"]}

    # Define model forcing variogram parameters
    OK = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model=model_name,
        variogram_parameters=variogram_parameters,
        verbose=False,
        enable_plotting=False,
    )

    # Predict on grid
    z, ss = OK.execute("grid", gridx, gridy)

    return z, OK
