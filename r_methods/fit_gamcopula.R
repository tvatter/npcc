#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript fit_gamcopula.R train.csv eval_grid.csv pred.csv")
}

train_file <- args[1]
eval_file  <- args[2]
out_file   <- args[3]

suppressPackageStartupMessages({
  library(dplyr)
  library(gamCopula)
})

trapz1 <- function(x, y) {
  if (length(x) < 2) return(0)
  sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

# h1 = d/du1 C(u1, u2 | x) = integral_0^{u2} c(u1, t | x) dt
h1_from_pdf <- function(fit_res, u1_target, u2_target, x0,
                        eps_int = 1e-3, n_int = 200) {
  upper <- max(u2_target, eps_int)
  grid <- seq(eps_int, upper, length.out = n_int)

  nd <- data.frame(
    u1 = rep(u1_target, length(grid)),
    u2 = grid,
    x  = rep(x0, length(grid))
  )
  pdf_vals <- as.numeric(gamBiCopPDF(fit_res, newdata = nd))
  trapz1(grid, pdf_vals)
}

# h2 = d/du2 C(u1, u2 | x) = integral_0^{u1} c(s, u2 | x) ds
h2_from_pdf <- function(fit_res, u1_target, u2_target, x0,
                        eps_int = 1e-3, n_int = 200) {
  upper <- max(u1_target, eps_int)
  grid <- seq(eps_int, upper, length.out = n_int)

  nd <- data.frame(
    u1 = grid,
    u2 = rep(u2_target, length(grid)),
    x  = rep(x0, length(grid))
  )
  pdf_vals <- as.numeric(gamBiCopPDF(fit_res, newdata = nd))
  trapz1(grid, pdf_vals)
}

train_df <- read.csv(train_file)
eval_df  <- read.csv(eval_file)

t_fit <- system.time({
  fit <- gamBiCopSelect(
    udata = train_df[, c("u1", "u2")],
    smooth.covs = data.frame(x = train_df[, "x"])
  )
})[["elapsed"]]

nd <- data.frame(u1 = eval_df$u1, u2 = eval_df$u2, x = eval_df$x)

t_pdf <- system.time({
  pdf_hat <- as.numeric(gamBiCopPDF(fit$res, newdata = nd))
})[["elapsed"]]

t_tau <- system.time({
  tau_hat <- as.numeric(
    gamBiCopPredict(fit$res, newdata = nd, target = "tau")$tau
  )
})[["elapsed"]]

t_h1 <- system.time({
  h1_hat <- mapply(
    FUN = function(u1, u2, x) h1_from_pdf(fit$res, u1, u2, x),
    u1 = eval_df$u1, u2 = eval_df$u2, x = eval_df$x
  )
})[["elapsed"]]

t_h2 <- system.time({
  h2_hat <- mapply(
    FUN = function(u1, u2, x) h2_from_pdf(fit$res, u1, u2, x),
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