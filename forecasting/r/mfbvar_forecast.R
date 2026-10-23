# MF-BVAR real-time forecasts for a list of forecast origins.
# Called from forecasting/mfbvar.py, one call = one specification (train start, variables, features type,
# availability) and several origins:
#
#     Rscript forecasting/r/mfbvar_forecast.R <task_dir>
#
# <task_dir> (written by Python):
#     task.json      settings + origins
#     monthly.csv    date | monthly features (already transformed: log d12 or its rolling mean)
#     quarterly.csv  date | targets (log d4), quarter start dates
# Results are appended after every origin, so a crash of R loses only the origin being estimated:
#     forecasts.csv  origin | target | target_date | pred | lower | upper
#     status.csv     origin | status (ok / error) | attempts | seconds | message
#
# Information set at origin T (quarter start) with availability a, the same as for the Python models:
#     monthly data up to month a of quarter T, targets up to quarter T - 1.

suppressPackageStartupMessages({
  library(mfbvar)
  library(jsonlite)
  library(lubridate)
  library(zoo)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript mfbvar_forecast.R <task_dir>")
task_dir <- args[1]

task <- fromJSON(file.path(task_dir, "task.json"))
monthly <- read.csv(file.path(task_dir, "monthly.csv"), check.names = FALSE)
monthly$date <- as.Date(monthly$date)
quarterly <- read.csv(file.path(task_dir, "quarterly.csv"), check.names = FALSE)
quarterly$date <- as.Date(quarterly$date)

sample_start <- as.Date(task$sample_start)
targets <- task$targets
forecasts_path <- file.path(task_dir, "forecasts.csv")
status_path <- file.path(task_dir, "status.csv")


append_csv <- function(frame, path) {
  exists <- file.exists(path)
  write.table(frame, path, sep = ",", row.names = FALSE, col.names = !exists, append = exists, quote = TRUE)
}

to_ts <- function(values, start_date, frequency) {
  period <- if (frequency == 12) month(start_date) else quarter(start_date)
  ts(values, start = c(year(start_date), period), frequency = frequency)
}

# one series cut at its last observation, interior gaps interpolated (no look-ahead: done after the cut)
clean_series <- function(values, dates, frequency) {
  observed <- which(!is.na(values))
  if (length(observed) < task$min_observations) return(NULL)
  first <- min(observed)
  last <- max(observed)
  values <- zoo::na.approx(values[first:last], na.rm = FALSE)
  if (sd(values) < 1e-10) return(NULL)
  list(series = to_ts(values, dates[first], frequency), first = dates[first], last = dates[last])
}

monthly_data <- function(origin) {
  cut <- origin %m+% months(task$availability - 1)
  frame <- monthly[monthly$date >= sample_start & monthly$date <= cut, , drop = FALSE]
  stale_before <- cut %m-% months(task$max_staleness_months)
  result <- list()
  for (name in setdiff(names(frame), "date")) {
    cleaned <- clean_series(frame[[name]], frame$date, 12)
    # mfbvar needs every monthly series to start at the first row; series that stopped long ago are useless
    if (is.null(cleaned) || cleaned$first != sample_start || cleaned$last < stale_before) next
    result[[name]] <- cleaned$series
  }
  if (length(result) == 0) stop("no monthly series with data at this origin")
  result
}

quarterly_data <- function(origin) {
  frame <- quarterly[quarterly$date >= sample_start & quarterly$date < origin, , drop = FALSE]
  result <- list()
  for (name in targets) {
    cleaned <- clean_series(frame[[name]], frame$date, 4)
    if (is.null(cleaned)) stop(sprintf("target %s has too few observations", name))
    result[[name]] <- cleaned$series
  }
  result
}

estimate <- function(origin, seed) {
  set.seed(seed)
  y <- c(monthly_data(origin), quarterly_data(origin))  # monthly first, then quarterly (mfbvar requirement)
  prior <- set_prior(
    Y = y,
    aggregation = "average",
    n_lags = task$n_lags,
    n_fcst = task$n_fcst_months,
    n_reps = task$n_reps,
    n_burnin = task$n_burnin,
    prior_Pi_AR1 = task$prior_Pi_AR1,
    lambda1 = task$lambda1,
    lambda2 = task$lambda2,
    lambda3 = task$lambda3,
    lambda4 = task$lambda4,
    check_roots = task$check_roots
  )
  model <- estimate_mfbvar(prior, prior = task$prior, variance = task$variance)
  fcst <- as.data.frame(predict(model, aggregate_fcst = TRUE, pred_bands = task$pred_bands))
  fcst <- fcst[fcst$variable %in% targets, , drop = FALSE]
  # quarterly forecasts are dated by the last month of the quarter -> quarter start
  quarter_start <- floor_date(as.Date(fcst$fcst_date), unit = "quarter")
  out <- data.frame(
    origin = as.character(origin),
    target = fcst$variable,
    target_date = as.character(quarter_start),
    pred = fcst$median,
    lower = fcst$lower,
    upper = fcst$upper,
    stringsAsFactors = FALSE
  )
  out <- out[quarter_start >= origin, , drop = FALSE]
  if (nrow(out) == 0 || any(!is.finite(out$pred))) stop("no finite forecasts")
  out
}

for (i in seq_along(task$origins)) {
  origin <- as.Date(task$origins[i])
  started <- Sys.time()
  result <- NULL
  message_text <- ""
  attempt <- 0
  # MCMC problems (e.g. no stable draw, singular matrices) are often seed specific -> retry with a new seed
  while (is.null(result) && attempt < task$max_attempts) {
    attempt <- attempt + 1
    seed <- task$seed + as.integer(origin) + 1000L * (attempt - 1L)
    result <- tryCatch(
      suppressWarnings(estimate(origin, seed)),
      error = function(e) {
        message_text <<- conditionMessage(e)
        NULL
      }
    )
  }
  seconds <- round(as.numeric(difftime(Sys.time(), started, units = "secs")), 1)
  if (!is.null(result)) {
    append_csv(result, forecasts_path)
    append_csv(data.frame(origin = as.character(origin), status = "ok", attempts = attempt,
                          seconds = seconds, message = ""), status_path)
  } else {
    append_csv(data.frame(origin = as.character(origin), status = "error", attempts = attempt,
                          seconds = seconds, message = gsub("[\r\n]+", " ", message_text)), status_path)
  }
  message(sprintf("[mfbvar] origin %s: %s in %.1fs", origin, if (is.null(result)) "error" else "ok", seconds))
}
