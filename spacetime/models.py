"""
Functions to run kriging or GP regression in 1D and 2D with GPyTorch, SciKit-Learn, PyKrige and GSTools.
"""

import numpy as np

import gpytorch
import torch
import sklearn.gaussian_process
from pykrige.ok import OrdinaryKriging
import gstools
from gstools import krige

from spacetime import conversions

####################################################################
# 1/ GPyTorch models and prediction fixing the covariance parameters
####################################################################
# Notes on correspondance with kriging:
# - a GPyTorch kernel is always a ScaleKernel (variogram sill) * BaseKernel (variogram form with range).
# So, in short:
# - sill = outputscale,
# - lengthscale = range (after some conversions),
# - nugget = noise_level (if homoscedastic).

def gpytorch_1d_model(train_x: torch.Tensor, train_y: torch.Tensor, base_kernel: gpytorch.kernels.Kernel, 
                      lengthscale: float, outputscale: float, noise: float, likelihood_missing=False, add_linear=False,
                      err: torch.Tensor = None, force_mean: float = None):
    """
    Define exact GPyTorch 1D model and fix covariance parameters.
    """

    # Exact GP model in 1D (see https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)
    class Exact1DModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact1DModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel())

            if add_linear:
                # self.covar_module += gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
                self.covar_module += gpytorch.kernels.LinearKernel()


        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    if likelihood_missing:
        likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs()
    elif err is not None:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=err**2)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = Exact1DModel(train_x, train_y, likelihood)

    if force_mean is not None:
        mean_y = torch.tensor(force_mean)
    else:
        mean_y = torch.mean(train_y)

    # Fix covariance parameters
    if not add_linear:
        hypers = {
            'mean_module.constant': mean_y,
            'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
            'covar_module.outputscale': torch.tensor(outputscale),
        }
        if err is None:
            hypers.update({'likelihood.noise_covar.noise': torch.tensor(noise)})

        model.initialize(**hypers)
    else:
        # Need to select the first kernel only to fix parameters when adding a Linear kernel
        if err is None:
            model.likelihood.noise_covar.noise = torch.tensor(noise).float()
        model.mean_module.constant = mean_y.float()
        model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale).float()
        model.covar_module.kernels[0].outputscale = torch.tensor(outputscale).float()

    return model

def gpytorch_2d_shared_model(train_x: torch.Tensor, train_y: torch.Tensor, base_kernel: gpytorch.kernels.Kernel, 
                             lengthscale: float, outputscale: float, noise: float):
    """
    Define exact GPyTorch 2D model and fix shared covariance parameters.
    """

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
        'likelihood.noise_covar.noise': torch.tensor(noise),
        'mean_module.constant': torch.mean(train_y),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(outputscale),
    }
    model.initialize(**hypers)

    return model

def gpytorch_predict_1d(variogram_model: dict[str, str | float], gridx: np.ndarray, data: np.ndarray,
                        err: np.ndarray = None, use_gpu=False, love=False, rank=100, likelihood_missing=False,
                        add_linear=False, force_mean=None):
    """
    Predict with GPyTorch 1D model on data.

    Returns means, sigmas and model.
    """

    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name, lengthscale, outputscale, noise_level = conversions.parse_covar_for_gp(variogram_model)

    # Convert x/y grid into
    grid_coords = torch.from_numpy(gridx)

    # X data is single coordinates (1 column)
    train_x = torch.from_numpy(data[:, 0])
    # Y data are values to predict (1 column)
    train_y = torch.from_numpy(data[:, 1])

    if err is not None:
        err = torch.from_numpy(err)

    base_kernel = getattr(gpytorch.kernels, base_kernel_name + "Kernel")  # GPyTorch adds "Kernel" to each name

    # Define model and fix covariance parameters
    model = gpytorch_1d_model(train_x=train_x, train_y=train_y, base_kernel=base_kernel,
                              lengthscale=lengthscale, outputscale=outputscale,
                              noise=noise_level, likelihood_missing=likelihood_missing, add_linear=add_linear,
                              err=err, force_mean=force_mean)
    likelihood = model.likelihood

    if use_gpu:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()
        grid_coords = grid_coords.cuda()

    # Predict
    model.eval()  # We go in eval mode without training
    likelihood.eval()
    if love:
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(rank):
            observed_pred = likelihood(model(grid_coords))  # And predict on the grid coordinates
    else:
        with torch.no_grad():
            observed_pred = likelihood(model(grid_coords))

    means = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

    if use_gpu:
        means = means.cpu()
        upper = upper.cpu()
        lower = lower.cpu()

    sigmas = (upper - means) / 2  # Returns two STD

    return means.numpy(), sigmas.numpy(), model


def gpytorch_predict_2d(variogram_model: dict[str, str | float], gridx: np.ndarray, gridy: np.ndarray, data: np.ndarray):
    """
    Predict with GPyTorch 2D model on data.

    Returns means, sigmas and model.
    """

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
                                     lengthscale=lengthscale, outputscale=outputscale,
                                     noise=np.sqrt(variogram_model["nugget"]))

    # Predict
    likelihood = model.likelihood
    model.eval()  # We go in eval mode without training
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(grid_coords))  # And predict on the grid coordinates

    means = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
    sigmas = (upper - means) / 2  # Returns two STD

    return means.numpy(), sigmas.numpy(), model


###############################################################
# 2/ SciKit-Learn GP model and prediction fixing the covariance
###############################################################

def sklearn_predict_1d(variogram_model: dict[str, str | float], gridx: np.ndarray, data: np.ndarray, use_gpu=False, love=False, rank=10):
    """
    Predict with SciKit-Learn 1D model on data.

    Returns means, sigmas and model.
    """

    # Convert parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Define kernel
    kernel = sklearn.gaussian_process.kernels.ConstantKernel(outputscale) * \
             getattr(sklearn.gaussian_process.kernels, base_kernel_name)(lengthscale) + \
             sklearn.gaussian_process.kernels.WhiteKernel(variogram_model["nugget"])

    # Define regressor, without optimizer to fix the parameters
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)

    # Transform y to be centered on the mean manually
    y_in = data[:, 1]
    mean_y = np.mean(y_in)
    y_in = y_in - mean_y

    gpr.fit(data[:, 0].reshape(-1, 1), y_in.reshape(-1, 1))

    y_pred, sigma = gpr.predict(gridx.reshape(-1, 1), return_std=True)
    y_pred = np.squeeze(y_pred)
    sigma = np.squeeze(sigma)

    # Transform back
    y_pred += mean_y

    return y_pred, sigma, gpr


def sklearn_predict_2d(variogram_model: dict[str, str | float], gridx: np.ndarray, gridy: np.ndarray, data: np.ndarray):
    """
    Predict with SciKit-Learn 2D model on data.

    Returns means, sigmas and model.
    """

    # Convert parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    # Define kernel
    kernel = sklearn.gaussian_process.kernels.ConstantKernel(outputscale) * \
             getattr(sklearn.gaussian_process.kernels, base_kernel_name)([lengthscale, lengthscale]) + \
             sklearn.gaussian_process.kernels.WhiteKernel(variogram_model["nugget"])

    # Define regressor, without optimizer to fix the parameters
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)

    # Transform y to be centered on the mean manually
    y_in = data[:, 2]
    mean_y = np.mean(y_in)
    y_in = y_in - mean_y

    gpr.fit(data[:, 0:2], y_in.reshape(-1, 1))

    x, y = np.meshgrid(gridx, gridy)
    grid_coords = torch.from_numpy(np.dstack((x.flatten(), y.flatten())).squeeze())

    y_pred, sigma = gpr.predict(grid_coords, return_std=True)
    y_pred = np.squeeze(y_pred)
    sigma = np.squeeze(sigma)

    # Transform back
    y_pred += mean_y

    return y_pred, sigma, gpr


################################################################
# 3/ PyKrige kriging models and prediction fixing the covariance
################################################################

def pykrige_predict_1d(variogram_model: dict[str, str | float], gridx: np.ndarray, data: np.ndarray):
    """
    Predict with PyKrige 1D model on data.

    Returns means, sigmas and model.
    """

    model_name = variogram_model["model_name"]
    variogram_parameters = {k: variogram_model[k] for k in ["psill", "range", "nugget"]}

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
    ss = np.squeeze(ss)

    return z, ss, OK


def pykrige_predict_2d(variogram_model: dict[str, str | float], gridx: np.ndarray, gridy: np.ndarray, data: np.ndarray):
    """
    Predict with PyKrige 2D model on data.

    Returns means, sigmas and model.
    """

    model_name = variogram_model["model_name"]
    variogram_parameters = {k: variogram_model[k] for k in ["psill", "range", "nugget"]}

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

    return z, ss, OK


################################################################
# 4/ GSTools kriging models and prediction fixing the covariance
################################################################

def gstools_predict_1d(variogram_model: dict[str, str | float], gridx: np.ndarray, data: np.ndarray):
    """
    Predict with GSTools 1D model on data.

    Returns means, sigmas and model.
    """

    model_name, range, sill = conversions.convert_kernel_pykrige_to_gstools(variogram_model)
    model = getattr(gstools, model_name)(dim=1, var=sill, len_scale=range, nugget=variogram_model["nugget"])

    k = krige.Ordinary(model, cond_pos=data[:, 0], cond_val=data[:, 1], exact=True)

    return k(gridx)[0], k(gridx)[1], k

def gstools_predict_2d(variogram_model: dict[str, str | float], gridx: np.ndarray, gridy: np.ndarray, data: np.ndarray):
    """
    Predict with GSTools 1D model on data.

    Returns means, sigmas and model.
    """

    model_name, range, sill = conversions.convert_kernel_pykrige_to_gstools(variogram_model)
    model = getattr(gstools, model_name)(dim=2, var=sill, len_scale=range, nugget=variogram_model["nugget"])

    k = krige.Ordinary(model, [data[:, 0], data[:, 1]], data[:, 2], exact=True)

    # Need to transpore the matrix for the first output of structured (the mean)
    return k.structured([gridx, gridy])[0].T, k.structured([gridx, gridy])[1].T, k

