#' @title Changepoint Detection Learner using PELT Algorithm
#' @name mlr_learners_changepoint.pelt
#'
#' @description
#' Changepoint detection via PELT (Pruned Exact Linear Time) algorithm
#' using [changepoint::cpt.mean()]. Wraps the algorithm as an
#' mlr3-compatible Learner.
#'
#' @export
LearnerChangepointPelt = R6::R6Class(
  "LearnerChangepointPelt",
  inherit = mlr3::LearnerRegr,

  public = list(

    #' @description Creates a new instance of this Learner.
    initialize = function() {
      param_set = paradox::ps(
        # Penalty value (higher = fewer changepoints)
        penalty = paradox::p_dbl(
          lower   = 0,
          upper   = Inf,
          default = 10,
          tags    = "train"
        ),
        # What to detect: mean, variance or meanvar
        method = paradox::p_fct(
          levels  = c("PELT", "BinSeg", "SegNeigh"),
          default = "PELT",
          tags    = "train"
        ),
        # Test statistic
        test_stat = paradox::p_fct(
          levels  = c("Normal", "CUSUM", "CSS", "Gamma", "Exponential"),
          default = "Normal",
          tags    = "train"
        )
      )

      super$initialize(
        id            = "regr.changepoint_pelt",
        feature_types = "numeric",
        predict_types = "response",
        param_set     = param_set,
        packages      = c("changepoint", "data.table"),
        label         = "PELT Changepoint Learner",
        man           = "mlr3changepoint::mlr_learners_changepoint.pelt"
      )
    }
  ),

  private = list(

    # ── TRAIN ─────
    .train = function(task) {
      data   = task$data(cols = task$feature_names)
      signal = as.numeric(data[[1]])

      pv        = self$param_set$values
      penalty   = pv$penalty   %||% 10
      method    = pv$method    %||% "PELT"
      test_stat = pv$test_stat %||% "Normal"

      # Fit PELT model
      fit = changepoint::cpt.mean(
        data      = signal,
        penalty   = "Manual",
        pen.value = penalty,
        method    = method,
        test.stat = test_stat
      )

      list(
        fit    = fit,
        signal = signal
      )
    },

    # ── PREDICT ───
    .predict = function(task) {
      model  = self$model
      fit    = model$fit
      signal = model$signal
      n      = length(signal)

      # Get changepoint positions
      cp     = changepoint::cpts(fit)
      breaks = c(0, cp, n)

      # Assign each observation its segment mean
      pred_vec = numeric(n)
      for (i in seq_len(length(breaks) - 1)) {
        idx              = (breaks[i] + 1):breaks[i + 1]
        pred_vec[idx]    = mean(signal[idx])
      }

      list(response = pred_vec)
    }
  )
)
