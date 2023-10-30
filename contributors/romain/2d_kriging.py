import matplotlib.pyplot as plt
import numpy as np

from pykrige.ok import OrdinaryKriging
import gpytorch
import torch

# 1/ Prepare synthetic data

# Sparse point data used as input
data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ]
)

# Grid for predicting the output
gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)

# Plot the point data on the grid
kwargs_cmap = {"vmin": 0, "vmax": 2, "cmap": "Spectral"}
plt.subplot(3, 3, 1)
plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], **kwargs_cmap)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.colorbar()
plt.title("Input points")
# plt.set_aspect('equal', adjustable='box')


# Define and convert variogram parameters (for both Kriging and GP!)
variogram_parameters = {'sill': 0.5, 'range': 2, 'nugget': 0}
model_name = "gaussian"

def convert_kernel_pykrige_to_gp(model_name, range, sill) -> tuple[str, float, float]:
    """

    :param model_name: Name of model in PyKrige.
    :param range: Range of model.
    :param sill: Sill of model.

    :return: Name of model in GPs, Lenghtscale, Outputscale
    """

    if model_name == "gaussian":
        # Lengthscale for gaussian in PyKrige: EXP(D**2 / A**2) with A = 4/7*R
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html#variogram-models

        # Lengthscale for RBF in GPs: EXP(-1/2 * D**2 / A**2)  (notice the 1/2 added at start of the EXP)
        # https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel

        # So 2A**2 = (4/7R)**2

        gp_model_name = "RBF"
        gp_lengthscale = range * (4/7) / np.sqrt(2)
        gp_outputscale = sill

    elif model_name == "exponential":
        # Lengthscale for exponential in PyKrige: EXP(D / A) with A = R/3
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html#variogram-models

        # Lengthscale for Matern with 1/2 smoothness in GPs (equivalent to exponential, see for instance https://scikit-gstat.readthedocs.io/en/latest/reference/models.html#matern-model):
        # EXP(-1/2 * D / A)  (notice the 1/2 added at start of the EXP)
        # https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel

        gp_model_name = "Matern"
        # Unsure about this one, but it looks like in SciKit-GStat the matern with scale 1/2 equals the exponential at same range,
        # and the exponential has R/3 and Matern R/2, so should be 2/3...
        gp_lengthscale = range * 2 / 3
        gp_outputscale = sill

    else:
        raise ValueError("model_name should be one of {}".format(["gaussian", "exponential"]))

    return gp_model_name, gp_lengthscale, gp_outputscale

sill = variogram_parameters["sill"]
range = variogram_parameters["range"]
gp_model_name, gp_lengthscale, gp_outputscale = convert_kernel_pykrige_to_gp("gaussian", range=range, sill=sill)

print("Variogram parameters:\n Model: {}, Sill: {}, Range: {}".format(model_name, sill, range))
print("Kernel parameters:\n Model: {}, Lengthscale: {}, Outputscale: {}".format(gp_model_name, gp_lengthscale, gp_outputscale))

# 2/ Run Ordinary Kriging
OK = OrdinaryKriging(
    data[:, 0],
    data[:, 1],
    data[:, 2],
    variogram_model=model_name,
    variogram_parameters=variogram_parameters,
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", gridx, gridy)

# kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
plt.subplot(3, 3, 3)
plt.imshow(z, **kwargs_cmap, origin="lower")
plt.colorbar()
plt.title("2D Ordinary Kriging")


# 3/ Run with GPyTorch
grid = torch.zeros(len(gridy), 2)
grid[:, 0] = torch.from_numpy(gridx)
grid[:, 1] = torch.from_numpy(gridy)

def get_2d_shared_model(train_x, train_y, lengthscale, outputscale):

    # Additive kernel
    class Shared2DGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Shared2DGP, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Shared2DGP(train_x, train_y, likelihood)

    # Single RBF kernel
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.001),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(outputscale),
    }
    model.initialize(**hypers)

    return model

def get_2d_additive_model(train_x, train_y, lengthscale, outputscale):

    # Additive kernel
    class Additive2DGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Additive2DGP, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(0,))) + \
                                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(1,)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Additive2DGP(train_x, train_y, likelihood)

    # Initialize parameters for additive kernel
    model.likelihood.noise_covar.noise = torch.tensor(0.0001),
    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[0].outputscale = torch.tensor(outputscale/2)
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[1].outputscale = torch.tensor(outputscale/2)

    return model

def get_2d_product_model(train_x, train_y, lengthscale, outputscale):

    # Additive kernel
    class Product2DGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(Product2DGP, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(0,))) * \
                                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(1,)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Product2DGP(train_x, train_y, likelihood)

    # Initialize parameters for additive kernel
    model.likelihood.noise_covar.noise = torch.tensor(0.0001),
    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[0].outputscale = torch.tensor(np.sqrt(outputscale))
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(lengthscale)
    model.covar_module.kernels[1].outputscale = torch.tensor(np.sqrt(outputscale))

    return model

# train_x = gpytorch.utils.grid.create_data_from_grid(grid)
train_x = torch.from_numpy(data[:, 0:2])
train_y = torch.from_numpy(data[:, 2])


model1 = get_2d_shared_model(train_x=train_x, train_y=train_y, lengthscale=gp_lengthscale, outputscale=gp_outputscale)
model2 = get_2d_additive_model(train_x=train_x, train_y=train_y, lengthscale=gp_lengthscale, outputscale=gp_outputscale)
model3 = get_2d_product_model(train_x=train_x, train_y=train_y, lengthscale=gp_lengthscale, outputscale=gp_outputscale)

list_models = [model1, model2, model3]
model_type = ["shared", "additive", "product"]
likelihood = gpytorch.likelihoods.GaussianLikelihood()

for i in np.arange(len(list_models)):

    model = list_models[i]
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        x, y = np.meshgrid(gridx, gridy)
        grid_coords = torch.from_numpy(np.dstack((x.flatten(), y.flatten())).squeeze())
        observed_pred = likelihood(model(grid_coords))


    means = observed_pred.mean

    plt.subplot(3, 3, 4 + i)
    plt.imshow(means.reshape((len(gridx), len(gridy))), **kwargs_cmap, origin="lower")
    plt.colorbar()
    plt.title("2D GPyTorch: "+model_type[i])

plt.tight_layout()
plt.show()
