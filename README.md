# spacetime-elevation

## End-of-hackweek summary

### Outcomes

This project of the [GeoSMART Hackweek](https://github.com/geo-smart) (initial pitch below) resulted in several investigations on Gaussian Process regression for big geospatial data, that may develop into discussions/PRs to upstream packages such as [GPyTorch](https://gpytorch.ai/) and [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/) and/or into a consolidation of advances in the package [GTSA](https://github.com/friedrichknuth/gtsa).
More specifically, the main perspectives related to the work done during the hackweek are to:
1. Ensure consistency of Gaussian Process regression between geostatistical kriging packages (e.g., [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/)) and machine-learning packages (e.g., [GPyTorch](https://gpytorch.ai/), [SciKit-Learn](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr)),
2. Add plotting tools and Leave-One-Out Cross-Validation for 1- or 2-D kernels in machine-learning Gaussian Process packages (learning from good practices in geostats),
3. Provide a scalable GP regression implementations in **space**, **time** and **space+time** for big geospatial data into a single package like [GTSA](https://github.com/friedrichknuth/gtsa) (due to the complexity and range of tools necessary: Rioxarray, Geocube, Dask, Rechuncker, Xbatcher, GPyTorch), and supporting adaptive chunks based on estimated kernel lengthscales.

More updates to come as things develop! :smile:

### Participants

* [Romain Hugonnet](https://github.com/rhugonnet) (lead)
* [Friedrich Knuth](https://github.com/friedrichknuth) (co-lead)
* [Keenan Ganz](https://github.com/s-kganz)
* [Morgan Sanger](https://github.com/sangermd)
* [Santiago Munevar](https://github.com/santimug)
* [George Lu](https://github.com/glugeorge)
* [Jessica Scheick](https://github.com/JessicaS11)
* [Emma Liu](https://github.com/EmmaLammE)
* [Emma Marshall](https://github.com/e-marshall)
* [Victor Devaux-Chupin](https://github.com/vdevauxchupin)
* [Ekin Ugurel](https://github.com/ekinugurel)
* [Tasha Snow](https://github.com/tsnow03)
* [Shiva Muruganandham](https://github.com/mshiv)
* [Imran Khan](https://github.com/imran-unh)

## Initial project pitch

Ready to elevate your spatiotemporal prediction skills? :space_invader: :satellite: :globe_with_meridians:

This is a project of the [GeoSMART Hackweek](https://github.com/geo-smart), taking place Oct 23-27 2023 at University of Washington.

Its focus is advancing spatiotemporal prediction of big data geospatial time series, in particular using **Gaussian Processes**. Our primary objective is to **estimate the continuous evolution of snow and ice covered surface elevations**, however the methods and tools we develop will be generic to any geospatial time series, whenever possible. **We welcome team members that would like to explore other datasets**, such as optical imagery, laser altimetry, synthetic aperture radar or temperature grids!

Part of this effort will include further development of the package Geospatial Time Series Analysis, [GTSA](https://github.com/friedrichknuth/gtsa), that provides routines for time-stacking and fitting geospatial gridded datasets out-of-memory, and possibly other upstream packages such as [RioXarray](https://github.com/corteva/rioxarray) and [xDEM](https://github.com/GlacioHack/xdem).

*Graphic from GTSA:*

<img src="https://github.com/friedrichknuth/gtsa-data/blob/main/img/stacking-light.png?raw=true#gh-light-mode-only" align="center" width="480px">
<img src="https://github.com/friedrichknuth/gtsa-data/blob/main/img/stacking-dark.png?raw=true#gh-dark-mode-only" align="center" width="480px">

## Spatiotemporal prediction of surface elevation changes, and more

### Summary

This projects aims to predict continuous spatiotemporal estimates from spatially and temporally sparse measurements.

It is primarily a software and data science method-oriented project, with the following three points of focus (decreasing order of envisioned work):
1. **Software development:** Develop a core Python package for scalable 3D (2D space + 1D time) geospatial analysis, building on [GTSA](https://github.com/friedrichknuth/gtsa).
2. **Data science method:** Practice the use of spatiotemporal prediction methods, in particular Gaussian Processes, for big remote sensing data.
3. **Applications:** Apply to glacier elevation changes, or snow depth, or more.

Tools that will be used: [Xarray](https://xarray.dev/), [Dask](https://docs.dask.org/en/stable/), [RioXarray](https://corteva.github.io/rioxarray/html/index.html), [GPyTorch](https://gpytorch.ai/).

### The problem

Observational data in Earth system science, whether ground or remote-sensing-based, is inherently sparse in space and time (e.g., point ground stations, fixed satellite footprint and revisit time).
For climate variables such as glaciers and seasonal snow that have substantial seasonal and regional variabilities, it is therefore difficult to reconcile observations between sites and time periods. This limitation largely hampers estimations of past changes (e.g., glacier mass changes, seasonal snow water equivalent) and their ingestion into models for predictions. 

### Goals

We identify two short-term goals (doable within the Hackweek timespan):

- **Start-up the development of a package on geospatial time series analysis** for 3-D space-time arrays which allows to apply existing methods in a scalable manner for georeferenced data,
- **Constrain the covariance of glacier or snow elevations** to correctly understand and apply Gaussian Process regression.

And two long-term goals (extending after the Hackweek):

- **Reach a stable version of a tested, documented and open source package on geospatial time series analysis**,
- **Publish a comparative study on the performance of spatio-temporal fitting methods** (parametric, non-parametric, physically-informed) for surface elevation.

### Background on proposed methods

Gaussian Processes are a promising avenue in non-parametric statistical modelling as, by learning the data covariance structure, they can provide a "best-unbiased estimator" for a specific problem using only the data itself. Gaussian Processes have the significant advantage of being independent of any physical assumptions (as in physically-based modelling) or parametrization (for other types of statistical modelling). 
Moreover, by learning the data covariance, Gaussian Process methods generally have the ability to predict reliable errors along their mean estimates. 

There is a lot of overlap between Gaussian Processes and geostatistics, as simple kriging is essentially another name for the same concept as Gaussian Processes. However, the generalization brought by Gaussian Processes to other fields has accelerated related research, in particular in terms of computational efficiency. With this aspect in mind, Gaussian Processes are now better adapted to the application of big data problems.

### Background on proposed tools

Based on the above, for computational efficiency, we would utilize Gaussian Processes packages. For scaling, it is best to compute on the GPU, which is integrated in [GPyTorch](https://gpytorch.ai/). 
In order to perform out-of-memory computations on large georeferenced datasets, we would combine [Xarray](https://xarray.dev/), [Dask](https://docs.dask.org/en/stable/) and [RioXarray](https://corteva.github.io/rioxarray/html/index.html).

To this end, **we aim to use and build upon the existing toolset in the Geospatial Time Series Analysis (GTSA) package**: https://github.com/friedrichknuth/gtsa.

### Data

Analysis ready dataset of:
1. **Historical (~1930s-1990s) photogrammetric DEMs in CONUS** stacked as zarr file and chunked along time dimension,
2. **Modern (2000s-2020s) ASTER and WorldView DEMs worldwide** stacked as zarr file and chunked along time dimension.
3. **Glacier outlines shapefiles.**

Available via AWS S3.

### Additional resources or background reading

**Reading and learning:**
- Interactive visualization of Gaussian Processes: http://www.infinitecuriosity.org/vizgp/,
- Scikit-learn examples: https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr.

**Video:**
- Application to glacier elevation changes using ICESat-2 (by Alex Gardner): https://www.youtube.com/watch?v=nhmREuVOWXg&t=1079.

**Code examples:**
- Application to temporal prediction of ASTER glacier elevation in **pyddem**: https://pyddem.readthedocs.io/en/v0.1.1/fitting.html#gaussian-process-regression-gpr,
- Application to historical glacier elevation: https://github.com/friedrichknuth/gtsa/blob/main/notebooks/processing/02_time_series_computations.ipynb.

**Publications:**
- Application to global glacier elevation changes with ASTER (Hugonnet et al., 2021): https://www.nature.com/articles/s41586-021-03436-z,
- Application to Icelandic glacier elevation with multiple high-res sensors (Bernat et al., 2023): https://dumas.ccsd.cnrs.fr/dumas-03772002.

### Tasks

In construction
