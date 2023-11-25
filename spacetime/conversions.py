"""
Functions to convert covariance between kriging and GP packages.
"""
import numpy as np


def convert_kernel_pykrige_to_gp(variogram_model: dict[str | float | str]) -> tuple[str, float, float]:
    """
    Convert PyKrige variogram to kernel (GPyTorch or SciKit-Learn).

    :param variogram_model: Dictionary with "model_name", "range" and "psill" corresponding to PyKrige definitions.
        See https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html.

    :return: Name of GPyTorch or SciKit-Learn GP model, Lenghtscale, Outputscale.
    """

    model_name = variogram_model["model_name"]
    psill = variogram_model["psill"]
    range = variogram_model["range"]

    if model_name == "gaussian":
        # Lengthscale for gaussian in PyKrige: EXP(D**2 / A**2) with A = 4/7*R
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html#variogram-models

        # Lengthscale for RBF in GPs: EXP(-1/2 * D**2 / A**2)  (notice the 1/2 added at start of the EXP)
        # https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel

        # So 2A**2 = (4/7R)**2

        gp_model_name = "RBF"
        gp_lengthscale = range * (4/7) / np.sqrt(2)
        gp_outputscale = psill

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
        gp_outputscale = psill

    else:
        raise ValueError("model_name should be one of {}".format(["gaussian", "exponential"]))

    return gp_model_name, gp_lengthscale, gp_outputscale



def convert_kernel_pykrige_to_gstools(variogram_model: dict[str | float | str]) -> tuple[str, float, float]:
    """
    Convert PyKrige variogram to GSTools variogram.

    :param variogram_model: Dictionary with "model_name", "range" and "psill" corresponding to PyKrige definitions.
        See https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html.

    :return: Name of GSTools model, Range, Sill.
    """

    model_name = variogram_model["model_name"]
    psill = variogram_model["psill"]
    range = variogram_model["range"]

    if model_name == "gaussian":
        # Lengthscale for gaussian in PyKrige: EXP(D**2 / A**2) with A = 4/7*R
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html#variogram-models

        # Lengthscale for RBF in GSTools: EXP(-1/2 * S**2 * D**2 / A**2)  with S=sqrt(pi)/2
        # https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.covmodel.Gaussian.html#gstools-covmodel-gaussian

        # So 2A**2 = (4/7R)**2

        gstools_model_name = "Gaussian"
        gstools_range = range * (4/7) * np.sqrt(np.pi) / 2
        gstools_psill = psill

    elif model_name == "exponential":
        # Lengthscale for exponential in PyKrige: EXP(-D / A) with A = R/3
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html#variogram-models

        # Lengthscale for GSTools: EXP(-S * D/A) with S = 1
        # https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.covmodel.Exponential.html

        gstools_model_name = "Exponential"
        # Unsure about this one, but it looks like in SciKit-GStat the matern with scale 1/2 equals the exponential at same range,
        # and the exponential has R/3 and Matern R/2, so should be 2/3...
        gstools_range = range * 3
        gstools_psill = psill

    else:
        raise ValueError("model_name should be one of {}".format(["gaussian", "exponential"]))

    return gstools_model_name, gstools_range, gstools_psill