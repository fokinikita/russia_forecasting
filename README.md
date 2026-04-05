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

   poetry run python main.py```

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

The `mfbvar` package has been removed from CRAN, but its archive is
included in the repository.
Alternatively, you can download it here:
https://cran.r-project.org/src/contrib/Archive/mfbvar/

File: mfbvar_0.5.6.tar.gz

Install via:

```install.packages("path_to_tar.gz", repos = NULL, type = "source")```

If the package is successfully installed, you can run the entire
`mfbvar.R` script
in parallel with `main.py` in Python.

You can also skip running MFBVAR. In that case, metrics for this model
will not be calculated.
Then in `main.py`, change:

```run_metrics(calculate_mfbvar=True)```

to:

```run_metrics(calculate_mfbvar=False)```

------------------------------------------------------------------------

# Running & Reproducibility

## Option 1 --- Precomputed forecasts

Simply evaluate metrics using precomputed forecasts stored in the
`preds` folder.

Use the notebook:

```metrics.ipynb```

------------------------------------------------------------------------

## Option 2 --- Full reproducible run

A fully reproducible run (training all models locally) can be done via
Poetry:

```poetry run python main.py```

This ensures consistent versions of all libraries.

You can also run:

```python main.py```

but results may differ slightly due to library versions, even though `random_seed` is fixed everywhere.

------------------------------------------------------------------------

## Option 3 --- Notebook

Use the notebook:

```metrics.ipynb```

It includes: - evaluation metrics
- test sets of length 12 and 24
- precomputed forecasts committed to the repository in `.csv` format

------------------------------------------------------------------------

## Notebook Environment Setup

It is recommended to use a separate environment via Poetry.

Install kernel:

```poetry add ipykernel```

```poetry run python -m ipykernel install --user --name=russia_forecasting --display-name "Python (russia_forecasting)"```

```poetry run jupyter notebook```

Then open `metrics.ipynb` and select the kernel:

```Python (russia_forecasting)```

