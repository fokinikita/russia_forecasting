# uncommetd those line if you dont have this packages

# install_version("readxl", version = "1.3.1", repos = "https://cran.r-project.org")
# install_version("writexl", version = "1.4.0", repos = "https://cran.r-project.org")
# install_version("lubridate", version = "1.9.0", repos = "https://cran.r-project.org")
# install_version("zoo", version = "1.8-11", repos = "https://cran.r-project.org")
# install_version("dplyr", version = "1.0.10", repos = "https://cran.r-project.org")
# install_version("data.table", version = "1.14.0", repos = "https://cran.r-project.org")

library(readxl)
library(writexl)
library(lubridate)
library(zoo)
library(dplyr)
library(data.table)

cd = "C:/Users/pc/Desktop/giga_data" # insert your path with this script and data here
setwd(cd)

# because the mfbvar package was deleted from CRAN
# we are installing archive, you can found it in the repo mfbvar_0.5.6.tar.gz
# Author's Github: https://github.com/ankargren/mfbvar
#install.packages("mfbvar_0.5.4.tar.gz", repos = NULL, type = "source")

library(mfbvar)
# Then just run all the code below, to get forecasts for further metrics calculations

START_YEAR = 2001
TEST_LEN = 12
p = 3
n_reps = 500

MAX_AVAILABILITY = 3
HORIZON = 6
MAX_HORIZON = 18

gamma= 0.68
alpha = 1 - gamma

prior_Pi_AR1 = 0.7
lambda1 = 0.1 
lambda2 = 0.5 
lambda3 = 1
lambda4 = 100

targets <- c("gdp_log_d4", "cons_log_d4", "inv_log_d4", "inv_cap_log_d4")

dataq = read.csv('data/giga_data_mfbvar_q.csv')
dataq$date <- as.Date(dataq$date)
dataq <- dataq[, colSums(is.na(dataq)) <= 4]
dataq <- na.omit(dataq)
dataq <- dataq %>%
  filter(date >= as.Date(paste0(START_YEAR, "-01-01")))
last_q_date <- max(dataq$date)
last_month_of_quarter <- last_q_date %m+% months(2)

datam = read.csv('data/giga_data_mfbvar_m.csv')
datam$date <- as.Date(datam$date)
datam <- datam[, colSums(is.na(datam)) <= 12]
datam <- na.omit(datam)
datam <- datam %>%
  filter(date >= as.Date(paste0(START_YEAR, "-01-01")))

datam <- datam %>% filter(date <= last_month_of_quarter)

all_results <- list()

for (avaliability in seq(1, MAX_AVAILABILITY)) {
  for (test_point in seq(1, TEST_LEN + HORIZON)) {

    message(sprintf("Evaluating model: availability=%d, test_point=%d",
                    avaliability, test_point))

    n <- nrow(dataq)
    train_q <- dataq[1:(n - test_point - 1), ]

    n_m <- nrow(datam)
    train_m <- datam[1:(n_m - test_point*3), ]
    nt_m <- nrow(train_m)

    if (avaliability < 3) {
      missing_months <- (3 - avaliability)
      train_m[(nt_m - missing_months + 1): (nt_m), colnames(datam) != "date"] <- NA
    }

    data_list <- list()

    for(j in colnames(datam)) {
      if(j != "date"){
        data_list[[j]] <- ts(
          na.omit(train_m[,j]),
          start = c(START_YEAR, 1),
          frequency = 12
        )
      }
    }

    data_list[['gdp_log_d4']] <- ts(
      train_q['gdp_log_d4'],
      start = c(START_YEAR, 1),
      frequency = 4
    )
    data_list[['cons_log_d4']] <- ts(
      train_q['cons_log_d4'],
      start = c(START_YEAR, 1),
      frequency = 4
    )
    data_list[['inv_log_d4']] <- ts(
      train_q['inv_log_d4'],
      start = c(START_YEAR, 1),
      frequency = 4
    )
    data_list[['inv_cap_log_d4']] <- ts(
      train_q['inv_cap_log_d4'],
      start = c(START_YEAR, 1),
      frequency = 4
    )

    tryCatch({

      prior <- set_prior(
        Y = data_list,
        n_lags = p,
        n_fcst = MAX_HORIZON,
        n_reps = n_reps,
        prior_Pi_AR1 = prior_Pi_AR1,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda3 = lambda3,
        lambda4 = lambda4,
        aggregation = "average",
        check_roots = TRUE
      )

      model <- estimate_mfbvar(prior, prior = "minn", variance = "iw")
      forecasts <- predict(model, aggregate_fcst = TRUE, pred_bands = gamma)

      result <- rbindlist(
        lapply(targets, function(v) {
          df <- forecasts[forecasts$variable == v, c("fcst_date", "median", "variable")][1:HORIZON, ]
          df$horizon <- seq_len(nrow(df))
          df$p <- p
          df$avaliability <- avaliability
          df
        }),
        use.names = TRUE
      )

      all_results[[length(all_results) + 1]] <- result

    }, error = function(e) {
      message(sprintf("Error at avaliability=%d, test_point=%d, skipped",
                      avaliability, test_point))
    })

  }
}

final_results <- rbindlist(all_results, use.names = TRUE)

fwrite(final_results, "preds/mfbvar_pred_test.csv")


avaliability <- 1
test_point <- 1

train_q <- dataq[1:(nrow(dataq) - test_point - 1), ]
train_m <- datam[1:(nrow(datam) - test_point*3), ]

# Подготовка data_list
data_list <- list()
for(j in colnames(datam)) {
  if(j != "date") data_list[[j]] <- ts(na.omit(train_m[,j]), start = c(START_YEAR,1), frequency = 12)
}
data_list[['gdp_log_d4']] <- ts(train_q['gdp_log_d4'], start=c(START_YEAR,1), frequency=4)
# ... добавь остальные цели

# Попробовать запустить prior и модель
prior <- set_prior(
  Y = data_list,
  n_lags = p,
  n_fcst = MAX_HORIZON,
  n_reps = n_reps,
  prior_Pi_AR1 = prior_Pi_AR1,
  lambda1 = lambda1,
  lambda2 = lambda2,
  lambda3 = lambda3,
  lambda4 = lambda4,
  aggregation = "average",
  check_roots = TRUE
)

model <- estimate_mfbvar(prior, prior="minn", variance="iw")
forecasts <- predict(model, aggregate_fcst=TRUE, pred_bands=gamma)


?mfbvar::set_prior()