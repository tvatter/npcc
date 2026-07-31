#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript fit_boostcopula.R train.csv eval_grid.csv pred.csv")
}

train_file <- args[1]
eval_file  <- args[2]
out_file   <- args[3]

suppressPackageStartupMessages({
  library(dplyr)
  library(boostCopula)
  library(VineCopula)
})

trapz1 <- function(x, y) {
  if (length(x) < 2) return(0)
  sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

getFams <- function(family) {
  if (family == 0) {
    fam <- 0
  } else if (family == 1) {
    fam <- 1
  } else if (family %in% c(301:304)) {
    fam <- as.numeric(rev(expand.grid(c(23, 33), c(3, 13)))[family - 300, ])
  } else if (family %in% c(401:404)) {
    fam <- as.numeric(rev(expand.grid(c(24, 34), c(4, 14)))[family - 400, ])
  } else {
    stop("Unknown family code: ", family)
  }
  fam
}

train_df <- read.csv(train_file)
eval_df  <- read.csv(eval_file)

t_fit <- system.time({
  fit <- boostBiCopSelect(
    formula = ~ .,
    U = train_df[, c("u1", "u2")],
    X = data.frame(x = train_df[, "x"]),
    familyset = NA
  )
})["elapsed"]

pdf_fun <- function(nd) {
  as.numeric(boostBiCopPDF(fit, nd[, c("u1", "u2")], data.frame(x = nd$x)))
}

# family bookkeeping, matching boostCopula internals
par <- predict(fit, type = "parameter")$parameter
fam <- getFams(as.numeric(fit$family))
family_vec <- rep(fam[1], length(par))
if (length(fam) == 2) family_vec[par < 0] <- fam[2]

tau_hat_fun <- function(nd) {
  par_x <- predict(fit, type = "parameter", newdata = data.frame(x = nd$x))$parameter
  fam <- getFams(as.numeric(fit$family))
  family_vec <- rep(fam[1], length(par_x))
  if (length(fam) == 2) family_vec[par_x < 0] <- fam[2]

  as.numeric(
    VineCopula::BiCopPar2Tau(
      family = family_vec,
      par = par_x,
      par2 = fit$par2,
      check.pars = FALSE
    )
  )
}

h1_from_pdf <- function(pdf_fun, u1, u2, x, eps_int = 1e-3, n_int = 200) {
  upper <- max(u2, eps_int)
  grid <- seq(eps_int, upper, length.out = n_int)

  nd <- data.frame(
    u1 = rep(u1, n_int),
    u2 = grid,
    x  = rep(x, n_int)
  )

  trapz1(grid, pdf_fun(nd))
}

h2_from_pdf <- function(pdf_fun, u1, u2, x, eps_int = 1e-3, n_int = 200) {
  upper <- max(u1, eps_int)
  grid <- seq(eps_int, upper, length.out = n_int)

  nd <- data.frame(
    u1 = grid,
    u2 = rep(u2, n_int),
    x  = rep(x, n_int)
  )

  trapz1(grid, pdf_fun(nd))
}



t_pdf <- system.time({
  pdf_hat <- pdf_fun(eval_df)
})[["elapsed"]]

t_tau <- system.time({
  tau_hat <- tau_hat_fun(eval_df)
})[["elapsed"]]

t_h1 <- system.time({
  h1_hat <- mapply(
    FUN = function(u1, u2, x) h1_from_pdf(pdf_fun, u1, u2, x),
    u1 = eval_df$u1, u2 = eval_df$u2, x = eval_df$x
  )
})[["elapsed"]]

t_h2 <- system.time({
  h2_hat <- mapply(
    FUN = function(u1, u2, x) h2_from_pdf(pdf_fun, u1, u2, x),
    u1 = eval_df$u1, u2 = eval_df$u2, x = eval_df$x
  )
})[["elapsed"]]

pred <- eval_df %>%
  mutate(
    pdf_hat = pdf_hat,
    tau_hat = tau_hat,
    h1_hat = h1_hat,
    h2_hat = h2_hat,
    fit_time = t_fit,
    pdf_time = t_pdf,
    tau_time = t_tau,
    h1_time = t_h1,
    h2_time = t_h2,
    total_estimator_time = t_fit + t_pdf + t_tau + t_h1 + t_h2
  ) %>%
  select(
    u1, u2, x,
    pdf_hat, tau_hat, h1_hat, h2_hat,
    fit_time, pdf_time, tau_time, h1_time, h2_time, total_estimator_time
  )

write.csv(pred, out_file, row.names = FALSE)