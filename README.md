# icy_spacetime_heights

This repository presents the `icy_spacetime_heights` project of the [GeoSMART Hackweek](https://github.com/geo-smart), taking place Oct 23-27 2023 at University of Washington.

Its efforts will focus on using and advancing the package Geospatial Time Series Analysis (GTSA) located here: [GTSA](https://github.com/friedrichknuth/gtsa).

## Spatiotemporal prediction of surface elevation changes on icy terrain from repeat DEMs

### Summary

This project focuses on predicting continuous surface elevation changes from spatially and temporally sparse elevation measurements.

It can be 
**Main focus 1 (data science method):** Gaussian Processes for big data.
**Main focus 2 (software development):** Develop a core Python package for scalable 3D (2D space + 1D time) geospatial analysis, building on [GTSA](https://github.com/friedrichknuth/gtsa).
**Secondary focus (applications):** Glacier elevation changes, snow depth.

Tools that will be used: [Xarray](https://xarray.dev/), [RioXarray](https://corteva.github.io/rioxarray/html/index.html), [GPyTorch](https://gpytorch.ai/).

### Collaborators

* [Romain Hugonnet](https://github.com/rhugonnet)
* [Friedrich Knuth](https://github.com/friedrichknuth)
* ...

### The problem

Observational data in Earth system science, whether ground or remote-sensing-based, is inherently sparse in space and time (e.g., point ground stations, fixed satellite footprint and revisit time).
For climate variables such as glaciers and seasonal snow that exhibit substantial seasonal and regional variabilities, it is therefore difficult to reconcile observations between sites and time period, and provide unbiased estimates. This limitation largely hampers estimations of past changes (e.g., glacier mass changes, seasonal snow water equivalent) and their ingestion into models for predictions. 

In this project, we aim to **develop efficient use of Gaussian Process regression on large datasets to predict spatiotemporally continuous surface elevation change from sparse elevation data**. 
While our project applies in particular to elevation data, it aims at **building a generic toolset for spatiotemporal methods on 3-D space-time arrays (2D space + 1D time)**.

### Goals

We identify two short-term goals (doable within the Hackweek timespan):

- **Start-up the development of a package on geospatial time series analysis** for 3-D space-time arrays which wraps existing methods in a scalable manner,
- **Constrain the form of the covariance for glacier and snow elevations**, to correctly understand and apply Gaussian Process regression.

And two long-term goals (extending after the Hackweek):

- **Reach a stable version of a tested, documented and open source package on geospatial time series analysis**,
- **Publish a comparative study on the performance of spatio-temporal fitting methods** (parametric, non-parametric, physically-informed) for surface elevation.

### Proposed methods

Gaussian Processes are a promising avenue in non-parametric statistical modelling as, by learning the data covariance structure, they can provide a non-parametric, "best-unbiased estimator" only based on the data itself for a specific problem. Gaussian Processes have the significant advantage of being independent of any physical assumptions (as in physically-based modelling) or parametrization (for other types of statistical modelling). 
Moreover, by learning the data covariance, Gaussian Process methods generally have the ability to predict reliable errors along their mean estimates. 

There is a lot of overlap between Gaussian Processes and geostatistics, as simple kriging is essentially another name for the same concept as Gaussian Processes. However, the generalization brought by Gaussian Processes to other fields has accelerated related research, in particular in terms of computational efficiency. With this aspect in mind, Gaussian Processes are now better adapted to the application of big data problems.

### Proposed tools

For computational efficiency, Gaussian Processes can be applied on GPU using [GPyTorch](https://gpytorch.ai/). 
However, they are not adapted specifically for use on geospatial data. We thus need to bring in [Xarray](https://xarray.dev/), and especially [RioXarray](https://corteva.github.io/rioxarray/html/index.html), to allow out-of-memory computations on large georeferenced datasets.

To combine those, **we aim to use and build upon the existing toolset in the Geospatial Time Series Analysis (GTSA) package**: https://github.com/friedrichknuth/gtsa.

### Data

Ongoing.

### Additional resources or background reading

**Reading:**
- Scikit-learn examples: https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr

**Code:**
- Application to temporal prediction of glacier elevation in **pyddem**: https://pyddem.readthedocs.io/en/v0.1.1/fitting.html#gaussian-process-regression-gpr. 
- 

**Publications:**
- Application to global glacier elevation changes with ASTER (Hugonnet et al., 2021): https://www.nature.com/articles/s41586-021-03436-z,
- Application to Icelandic glacier elevation with multiple high-res sensors (Bernad et al., 2023): https://dumas.ccsd.cnrs.fr/dumas-03772002.

**Video:**
- Application to glacier elevation changes using ICESat-2 (by Alex Gardner): https://www.youtube.com/watch?v=nhmREuVOWXg&t=1079.

### Tasks

What are the individual tasks or steps that need to be taken to achieve the project goals? Think about which tasks are dependent on prior tasks, or which tasks can be performed in parallel. This can help divide up project work among team members.

* Task 1 (all team members)
* Task 2
  * Task 2a (assigned to team member A)
  * Task 2b (assigned to team member B)
* Task 3
* ...
