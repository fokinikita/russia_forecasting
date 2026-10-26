# One image for the whole service: Airflow (own virtualenv) schedules the weekly run, every task calls
# the project code (/opt/venv) which uses R (Rscript) as a worker for MF-BVAR. See README, "Сервис".
#
#   docker compose up -d --build            # Airflow on http://<server>:8095, schedule Mon 05:00 MSK
#
#   docker build -t russia-forecasting .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#              -v "$PWD/dashboard:/app/dashboard" russia-forecasting
#
# R 4.2.2 on Ubuntu 22.04; rocker freezes CRAN to a Posit Package Manager snapshot of that
# R release (binary packages -> fast and reproducible builds).
FROM rocker/r-ver:4.2.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    RSCRIPT=Rscript \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/opt/python

# build tools for mfbvar / stochvol / RcppArmadillo (C++), BLAS/LAPACK headers, certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran make ca-certificates curl \
        libopenblas-dev liblapack-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- R part
WORKDIR /build
COPY docker/install_r_packages.R mfbvar_0.5.6.tar.gz ./
RUN Rscript install_r_packages.R mfbvar_0.5.6.tar.gz && rm -rf /build /tmp/Rtmp*

# ----------------------------------------------------------- Python part
# Python 3.12 (same as the development environment) via uv's standalone builds
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv \
    && uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && uv pip install --no-cache -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# --------------------------------------------------------------- Airflow
# separate virtualenv: Airflow pins its own dependency versions (constraints file), the models keep theirs
ARG AIRFLOW_VERSION=3.1.8
ENV AIRFLOW_HOME=/opt/airflow
RUN uv venv --python 3.12 /opt/airflow-venv     && uv pip install --no-cache --python /opt/airflow-venv/bin/python         "apache-airflow[amazon]==${AIRFLOW_VERSION}"         --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"     && mkdir -p /opt/airflow

# ------------------------------------------------------------------ code
WORKDIR /app
COPY . /app
RUN chmod -R a+rwX /app && mkdir -p /tmp/matplotlib && chmod 1777 /tmp/matplotlib

# fails the build early if Python or R dependencies are broken
RUN python -c "import catboost, ngboost, pytorch_tabnet, statsmodels, polars, torch, boto3, pdfplumber, service.cli; print('python ok')"     && /opt/airflow-venv/bin/python orchestration/dags/russia_forecasting.py && echo "dag ok" \
    && Rscript -e "suppressPackageStartupMessages(library(mfbvar)); cat('mfbvar', as.character(packageVersion('mfbvar')), 'ok\n')" \
    && python -m pytest tests -q -p no:cacheprovider

RUN chmod +x docker/airflow-entrypoint.sh
EXPOSE 8080
CMD ["docker/airflow-entrypoint.sh"]
