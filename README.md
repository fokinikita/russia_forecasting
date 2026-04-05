# Russian Forecasting: ML vs Classical Econometric Models

This project is dedicated to comparing **machine learning models** and
**classical econometric approaches** for forecasting key **macroeconomic
indicators of Russia**.

------------------------------------------------------------------------

## Plans

In the future, the project will include an automated data parser and
forecasting pipeline with prediction horizons of **1.5--2 years
ahead**.\
Currently, all computations are based on **manually collected
datasets**.

------------------------------------------------------------------------

## Models Used

### Machine Learning

-   CatBoost
-   NGBoost
-   TabNet

### Econometric Models

-   MFBVAR (Mixed-Frequency Bayesian VAR)
-   DFM (Dynamic Factor Model)
-   Naive benchmark (random walk / last observed value)

------------------------------------------------------------------------

# Installation

To install the project locally, create an empty directory and run the
following commands:

> Requires Python 3.12

    git clone https://github.com/fokinikita/russia_forecasting
    cd russia_forecasting
    poetry config virtualenvs.in-project true
    poetry env use 3.12
    poetry install

Run the project:

    poetry run python main.py

------------------------------------------------------------------------

# Data

All feature variables (**63 variables**) are described in
`var_naming.xlsx`. (in Russian)

To download raw data:

    poetry run python -m pipelines.download_prepared_raw_data

------------------------------------------------------------------------

## Target Variables

1.  GDP
2.  Household consumption
3.  Gross capital formation
4.  Gross fixed capital formation

------------------------------------------------------------------------

## Project Structure

    giga_data/
    ├── main.py
    ├── pipelines/
    ├── data/
    ├── models/
    ├── metrics/
    ├── config.py
    ├── pyproject.toml
    ├── poetry.lock
    └── README.md

------------------------------------------------------------------------

## MFBVAR in R

Download: https://cran.r-project.org/src/contrib/Archive/mfbvar/

Install:

    install.packages("path_to_tar.gz", repos = NULL, type = "source")

------------------------------------------------------------------------

# Running

    poetry run python main.py
