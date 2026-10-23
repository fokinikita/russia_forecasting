# R dependencies of forecasting/r/mfbvar_forecast.R.
# mfbvar 0.5.6 was archived on CRAN, it is installed from the tarball in the repo.
# RcppArmadillo and stochvol are pinned to versions mfbvar is known to compile and run with
# (the same as in the working Windows setup), so later releases cannot break the build.
#
#     Rscript docker/install_r_packages.R mfbvar_0.5.6.tar.gz

args <- commandArgs(trailingOnly = TRUE)
tarball <- if (length(args)) args[1] else "mfbvar_0.5.6.tar.gz"
archive <- "https://cloud.r-project.org"
ncpus <- max(1L, parallel::detectCores())

options(Ncpus = ncpus)
install.packages(c(
  "remotes", "jsonlite", "lubridate", "zoo", "dplyr", "magrittr", "tibble", "ggplot2",
  "GIGrvg", "coda", "Rcpp", "RcppProgress", "RcppParallel"
))

remotes::install_version("RcppArmadillo", version = "0.11.4.2.1", repos = archive, upgrade = "never")
remotes::install_version("stochvol", version = "3.2.0", repos = archive, upgrade = "never")

install.packages(tarball, repos = NULL, type = "source")

for (pkg in c("mfbvar", "jsonlite", "lubridate", "zoo")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("package ", pkg, " was not installed")
}
cat("RcppArmadillo", as.character(packageVersion("RcppArmadillo")),
    "| stochvol", as.character(packageVersion("stochvol")),
    "| mfbvar", as.character(packageVersion("mfbvar")), "\n")
