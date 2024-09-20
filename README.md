# Diffusion Models for Tabular and Time Series Bootcamp

This repository contains codes for diffusion models bootcamp.

## About Diffusion Models for Tabular and Time Series Bootcamp

Diffusion models are a class of generative models that have shown promising results in various domains, including image generation, language modeling, and time series forecasting. In this bootcamp, we will explore the application of diffusion models to tabular and time series data.

## Repository Structure

- **docs/**: Contains detailed documentation, additional resources, installation guides, and setup instructions that are not covered in this README.
- **reference_implementations/**: Reference Implementations are organized by topics. Each topic has its own directory containing codes, notebooks, and a README for guidance.
- **pyproject.toml**: The `pyproject.toml` file in this repository configures various build system requirements and dependencies, centralizing project settings in a standardized format.


### Reference Implementations Directory

Each topic within the bootcamp has a dedicated directory in the `reference_implementations/` directory. In each directory, there is a README file that provides an overview of the topic, prerequisites, and notebook descriptions.

Here is the list of the covered topics:
- Tabular Data
  - TabDDPM for Tabular Data Generation
  - TabSyn for Tabular Data Generation and Imputation
  - ClavaDDPM for Multi-relational Tabular Data Generation
- Time series
  - CSDI for Time Series Forecasting and Imputation
  - TS-Diff for Time Series Forecasting, Synthesizing and Refining other Forecasting Models

## Getting Started

To get started with this bootcamp:
1. Clone this repository to your machine.
2. Activate your python environment. If you are using Vector's cluster, you can activate it by refering to the `docs/technical_onboarding`, otherwise you can create a new environment using the following command:
```bash
pip install --upgrade pip poetry
poetry env use [name of your python] #python3.9
source $(poetry env info --path)/bin/activate
poetry install --with "synthcity, tabsyn, clavaddpm, csdi, tsdiff"
# If your system is not compatible with pykeops, you can uninstall it using the following command
pip uninstall pykeops 
# Install the kernel for jupyter (only need to do it once)
ipython kernel install --user --name=diffusion_models
```
3. Begin with each topic in the `reference_implementations/` directory, as guided by the README files.

## License
This project is licensed under the terms of the [LICENSE] file located in the root directory of this repository.

## Contribution
To get started with contributing to our project, please read our [CONTRIBUTING.md] guide.

## Contact Information

For more information or help with navigating this repository, please contact Vector AI ENG Team at sana.ayromlou@vectorinstitute.ai .

