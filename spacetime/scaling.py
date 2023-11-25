"""
Functions to scale computations across large data.
"""

import numpy as np

import gpytorch
import torch
import sklearn

from spacetime import conversions, models


# GPyTorch models useful for independent dimensional prediction

def gpytorch_2d_product_model(train_x, train_y, base_kernel1, base_kernel2,
                              lengthscale1, outputscale1, lengthscale2, outputscale2, noise):

    # Product 2D kernel
    class Exact2DProduct(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Exact2DProduct, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel1(active_dims=(0,))) * \
                                gpytorch.kernels.ScaleKernel(base_kernel2(active_dims=(1,)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Exact2DProduct(train_x, train_y, likelihood)

    # Initialize parameters for product kernel
    model.likelihood.noise_covar.noise = torch.tensor(noise).float()
    model.mean_module.constant = torch.mean(train_y).float()

    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale1).float()
    model.covar_module.kernels[0].outputscale = torch.tensor(outputscale1).float()
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(lengthscale2).float()
    model.covar_module.kernels[1].outputscale = torch.tensor(outputscale2).float()

    return model


def gpytorch_1d_batch_model(train_x, train_y, base_kernel, lengthscale, outputscale, noise, batch_size):

    # Batch 1D kernel
    class Batch1D(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Batch1D, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel(batch_shape=torch.Size([batch_size])), batch_shape=torch.Size([batch_size]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([batch_size]))
    model = Batch1D(train_x, train_y, likelihood)

    # Initialize parameters for product kernel
    model.likelihood.noise_covar.noise = torch.tensor(noise).float()
    model.mean_module.constant = torch.mean(train_y).float()
    model.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale).float()
    model.covar_module.outputscale = torch.tensor(outputscale).float()

    return model

# Routines to predict with independent dimensions

def loop_predict_1d(predict_fun, variogram_model, gridx, x_in, dc, use_gpu=False, love=False, rank=10):

    pred_y = np.zeros(shape=(len(gridx), *dc.shape[1:]))
    pred_sig = np.zeros(shape=(len(gridx), *dc.shape[1:]))
    for i in range(dc.shape[1]):

        # Stack the same 1D coordinate to the data, keeping only valid points
        data_1d = np.dstack((x_in, dc[:, i])).squeeze()
        ind_valid = np.isfinite(data_1d[:, 1])
        y, sig = predict_fun(variogram_model=variogram_model, gridx=gridx, data=data_1d[ind_valid, :], use_gpu=use_gpu, love=love, rank=rank)[:2]
        pred_y[:, i] = y
        pred_sig[:, i] = sig

    return pred_y, pred_sig

def gpytorch_predict_2d_product(variogram_model1, variogram_model2, gridx, gridy, data):

    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name1, lengthscale1, outputscale1 = conversions.convert_kernel_pykrige_to_gp(variogram_model1)
    base_kernel_name2, lengthscale2, outputscale2 = conversions.convert_kernel_pykrige_to_gp(variogram_model2)

    # Convert x/y grid into
    x, y = np.meshgrid(gridx, gridy)
    grid_coords = torch.from_numpy(np.dstack((x.flatten(), y.flatten())).squeeze()).float()

    # X data are coordinates (2 columns)
    train_x = torch.from_numpy(data[:, 0:2]).float()
    # Y data are values to predict (1 column)
    train_y = torch.from_numpy(data[:, 2]).float()

    base_kernel1 = getattr(gpytorch.kernels, base_kernel_name1 + "Kernel")  # GPyTorch adds "Kernel" to each name
    base_kernel2 = getattr(gpytorch.kernels, base_kernel_name2 + "Kernel")  # GPyTorch adds "Kernel" to each name

    # Define model and fix covariance parameters
    model = gpytorch_2d_product_model(train_x=train_x, train_y=train_y,
                                      base_kernel1=base_kernel1, lengthscale1=lengthscale1, outputscale1=outputscale1,
                                      base_kernel2=base_kernel2, lengthscale2=lengthscale2, outputscale2=outputscale2,
                                      noise=variogram_model1["nugget"])

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

def sklearn_predict_2d_product(variogram_model1, variogram_model2, gridx, gridy, data):

    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name1, lengthscale1, outputscale1 = conversions.convert_kernel_pykrige_to_gp(variogram_model1)
    base_kernel_name2, lengthscale2, outputscale2 = conversions.convert_kernel_pykrige_to_gp(variogram_model2)

    # Define kernel
    assert base_kernel_name1 == base_kernel_name2
    kernel = sklearn.gaussian_process.kernels.ConstantKernel(outputscale1) * \
             sklearn.gaussian_process.kernels.ConstantKernel(outputscale2) * \
             getattr(sklearn.gaussian_process.kernels, base_kernel_name1)([lengthscale1, lengthscale2]) + \
             sklearn.gaussian_process.kernels.WhiteKernel(variogram_model1["nugget"])

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


def gpytorch_predict_1d_batch(variogram_model, gridx, x_in, data_batch, use_gpu=False, love=False, rank=100):

    # Convert variogram form and parameters into base kernel form and parameters
    base_kernel_name, lengthscale, outputscale = conversions.convert_kernel_pykrige_to_gp(variogram_model)

    batch_size = data_batch.shape[0]

    # Convert x/y grid into
    grid_coords = torch.from_numpy(np.repeat(gridx.reshape((1, -1)), batch_size, axis=0)).reshape((batch_size, -1, 1))

    # X data is single coordinates (1 column)
    train_x = torch.from_numpy(x_in.reshape((batch_size, -1, 1)))
    # Y data are values to predict (1 column)
    train_y = torch.from_numpy(data_batch)

    base_kernel = getattr(gpytorch.kernels, base_kernel_name + "Kernel")  # GPyTorch adds "Kernel" to each name

    # Define model and fix covariance parameters
    model = gpytorch_1d_batch_model(train_x=train_x, train_y=train_y, base_kernel=base_kernel,
                                    lengthscale=lengthscale, outputscale=outputscale,
                                    noise=variogram_model["nugget"], batch_size=batch_size)
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