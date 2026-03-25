#' @title Changepoint Detection Learner using Binary Segmentation
#' @name mlr_learners_changepoint.binseg
#'
#' @description
#' Changepoint detection via binary segmentation using [binsegRcpp::binseg()].
#' Wraps the algorithm as an mlr3-compatible Learner.
#'
#' @export
LearnerChangepointBinSeg = R6::R6Class(
  "LearnerChangepointBinSeg",
  inherit = mlr3::LearnerRegr,

  public = list(

    #' @description Creates a new instance of this Learner.
    initialize = function() {
      # Define hyperparameters
      param_set = paradox::ps(
        # Number of segments to fit
        n_segments = paradox::p_int(
          lower   = 2L,
          upper   = Inf,
          default = 5L,
          tags    = "train"
        ),
        # Distribution/loss function
        distribution = paradox::p_fct(
          levels  = c("mean_norm", "variance", "meanvar_norm", "poisson"),
          default = "mean_norm",
          tags    = "train"
        )
      )

      super$initialize(
        id            = "regr.changepoint_binseg",
        feature_types = "numeric",
        predict_types = "response",
        param_set     = param_set,
        packages      = c("binsegRcpp", "data.table"),
        label         = "Binary Segmentation Changepoint Learner",
        man           = "mlr3changepoint::mlr_learners_changepoint.binseg"
      )
    }
  ),

  private = list(

    # ── TRAIN ───#
    .train = function(task) {
      # Get the signal (first feature column)
      data         = task$data(cols = task$feature_names)
      signal       = as.numeric(data[[1]])

      pv           = self$param_set$values
      n_seg        = pv$n_segments   %||% 5L
      distribution = pv$distribution %||% "mean_norm"

      # Fit binary segmentation model
      fit = binsegRcpp::binseg(
        distribution.str = distribution,
        data.vec         = signal,
        max.segments     = n_seg
      )

      # Store everything needed for prediction
      list(
        fit          = fit,
        signal       = signal,
        n_segments   = n_seg,
        distribution = distribution
      )
    },

    # ── PREDICT ───#
    .predict = function(task) {
      model  = self$model
      fit    = model$fit
      n_seg  = model$n_segments

      # Get segment coefficients
      coef_dt = data.table::as.data.table(coef(fit, n_seg))

      # Build a predicted mean signal (each obs gets its segment mean)
      signal   = model$signal
      pred_vec = numeric(length(signal))

      for (i in seq_len(nrow(coef_dt))) {
        start_i = if (i == 1) 1L else coef_dt$start[i]
        end_i   = coef_dt$end[i]
        pred_vec[start_i:end_i] = coef_dt$mean[i]
      }

      # mlr3 requires a PredictionRegr — return predicted values
      list(response = pred_vec)
    }
  )
)
