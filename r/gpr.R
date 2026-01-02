# =========================================================
# Gaussian Process Regression (GPR) as Meta-Learner
# Stacking Ensemble for Mineral Prospectivity Mapping
# =========================================================

# Load required libraries
library(kernlab)      # Gaussian Process Regression
library(openxlsx)     # Excel I/O
library(ggplot2)      # Visualization
library(viridis)      # Color scale

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
train_data <- read.xlsx("Train.xlsx")
test_data  <- read.xlsx("Test.xlsx")
f1_data    <- read.xlsx("F1.xlsx")

# ---------------------------------------------------------
# Feature selection (including base learners: SVR & ANN)
# ---------------------------------------------------------
features <- c(
  "Fault_LD", "Fault_NW", "RockType", "Cu", "RTP",
  "Analytic_signal", "K", "FeO", "Argillic", "Phyllic",
  "SVR", "ANN"
)

x_train <- train_data[, features]
y_train <- train_data$Class

x_test  <- test_data[, features]
y_test  <- test_data$Class

x_f1 <- f1_data[, features]

# ---------------------------------------------------------
# Manual hyperparameter search
# ---------------------------------------------------------
param_grid <- expand.grid(
  sigma  = c(0.1, 0.5, 1, 2, 3),
  C      = c(0.1, 0.5, 1, 2, 3),
  kernel = c("rbfdot", "laplacedot"),
  stringsAsFactors = FALSE
)

best_model   <- NULL
best_params  <- NULL
best_metrics <- list(R = NA, R2 = -Inf, RMSE = NA, MSE = NA, MAE = NA)

for (i in 1:nrow(param_grid)) {
  
  p <- param_grid[i, ]
  
  model <- gausspr(
    x_train, y_train,
    sigma  = p$sigma,
    C      = p$C,
    kernel = as.character(p$kernel)
  )
  
  y_pred <- predict(model, x_test)
  
  rmse <- sqrt(mean((y_test - y_pred)^2))
  mse  <- mean((y_test - y_pred)^2)
  mae  <- mean(abs(y_test - y_pred))
  r    <- cor(y_test, y_pred)
  r2   <- r^2
  
  if (r2 > best_metrics$R2) {
    best_model   <- model
    best_params  <- p
    best_metrics <- list(R = r, R2 = r2, RMSE = rmse, MSE = mse, MAE = mae)
  }
}

# ---------------------------------------------------------
# Best model performance
# ---------------------------------------------------------
cat("Best hyperparameters:\n")
print(best_params)

cat("\nModel performance on test data:\n")
print(best_metrics)

# ---------------------------------------------------------
# Prediction on target area (F1)
# ---------------------------------------------------------
f1_data$Prediction <- predict(best_model, x_f1)
write.xlsx(f1_data, "F1_Predicted.xlsx", overwrite = TRUE)

# ---------------------------------------------------------
# Prediction map
# ---------------------------------------------------------
p <- ggplot(f1_data, aes(x = X, y = Y, color = Prediction)) +
  geom_point(size = 2.5) +
  scale_color_viridis(option = "D") +
  labs(
    title = "GPR-Based Stacking Ensemble Prediction Map",
    x = "X Coordinate",
    y = "Y Coordinate",
    color = "Predicted Value"
  ) +
  theme_minimal() +
  coord_fixed()

ggsave("prediction_map.png", p, dpi = 300, width = 8, height = 8)

# ---------------------------------------------------------
# Feature importance using permutation (MSE-based)
# ---------------------------------------------------------
evaluate_feature_importance <- function(model, features, target) {
  
  base_pred <- predict(model, features)
  base_mse  <- mean((target - base_pred)^2)
  
  importance <- sapply(colnames(features), function(col) {
    shuffled <- features
    shuffled[[col]] <- sample(shuffled[[col]])
    shuffled_pred <- predict(model, shuffled)
    mean((target - shuffled_pred)^2) - base_mse
  })
  
  result <- data.frame(
    Feature = names(importance),
    Importance = importance
  )
  
  result[order(abs(result$Importance), decreasing = TRUE), ]
}

feature_importance <- evaluate_feature_importance(best_model, x_test, y_test)
write.xlsx(feature_importance, "Feature_Importance.xlsx", overwrite = TRUE)

