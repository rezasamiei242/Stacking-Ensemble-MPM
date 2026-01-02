# =========================================================
# Logistic Regression as Meta-Learner
# Stacking Ensemble for Mineral Prospectivity Mapping
# =========================================================

# ---------------------------------------------------------
# Load required libraries
# ---------------------------------------------------------
library(openxlsx)
library(ggplot2)
library(viridis)
library(caret)

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
train_data <- read.xlsx("Train.xlsx")
test_data  <- read.xlsx("Test.xlsx")
f1_data    <- read.xlsx("F1.xlsx")

# ---------------------------------------------------------
# Feature selection (including SVR & ANN predictions)
# ---------------------------------------------------------
features <- c(
  "Fault_LD", "Fault_NW", "RockType", "Cu", "RTP",
  "Analytic_signal", "K", "FeO", "Argillic", "Phyllic",
  "SVR", "ANN"
)

x_train <- train_data[, features]
y_train <- as.factor(train_data$Class)

x_test <- test_data[, features]
y_test <- as.factor(test_data$Class)

x_f1 <- f1_data[, features]

# ---------------------------------------------------------
# Logistic regression with Elastic Net regularization
# ---------------------------------------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE
)

param_grid <- expand.grid(
  alpha  = c(0, 0.5, 1),       # 0 = L2, 1 = L1, elastic net
  lambda = c(0.001, 0.01, 0.1, 1)
)

log_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = param_grid
)

# ---------------------------------------------------------
# Best hyperparameters
# ---------------------------------------------------------
cat("Best hyperparameters:\n")
print(log_model$bestTune)

# ---------------------------------------------------------
# Model evaluation on test data
# ---------------------------------------------------------
y_pred_prob <- predict(log_model, x_test, type = "prob")[, 2]
y_test_num  <- as.numeric(as.character(y_test))

rmse <- sqrt(mean((y_test_num - y_pred_prob)^2))
mse  <- mean((y_test_num - y_pred_prob)^2)
mae  <- mean(abs(y_test_num - y_pred_prob))
r    <- cor(y_test_num, y_pred_prob)
r2   <- r^2

cat("\nModel performance on test data:\n")
cat("R   =", r, "\n")
cat("R2  =", r2, "\n")
cat("RMSE=", rmse, "\n")
cat("MSE =", mse, "\n")
cat("MAE =", mae, "\n")

# ---------------------------------------------------------
# Prediction on target area (F1)
# ---------------------------------------------------------
f1_data$Prediction <- predict(log_model, x_f1, type = "prob")[, 2]
write.xlsx(f1_data, "F1_Predicted_Logistic.xlsx", overwrite = TRUE)

# ---------------------------------------------------------
# Prediction map
# ---------------------------------------------------------
p <- ggplot(f1_data, aes(x = X, y = Y, color = Prediction)) +
  geom_point(size = 2.5) +
  scale_color_viridis(option = "D") +
  labs(
    title = "Logistic Regression Stacking Ensemble Prediction Map",
    x = "X Coordinate",
    y = "Y Coordinate",
    color = "Predicted Probability"
  ) +
  theme_minimal() +
  coord_fixed()

ggsave("prediction_map_logistic.png", p, dpi = 300, width = 8, height = 8)

# ---------------------------------------------------------
# Feature importance using permutation (MSE-based)
# ---------------------------------------------------------
evaluate_feature_importance <- function(model, features, target) {
  
  base_prob <- predict(model, features, type = "prob")[, 2]
  base_mse  <- mean((as.numeric(as.character(target)) - base_prob)^2)
  
  importance <- sapply(colnames(features), function(col) {
    shuffled <- features
    shuffled[[col]] <- sample(shuffled[[col]])
    shuffled_prob <- predict(model, shuffled, type = "prob")[, 2]
    mean((as.numeric(as.character(target)) - shuffled_prob)^2) - base_mse
  })
  
  result <- data.frame(
    Feature = names(importance),
    Importance = importance
  )
  
  result[order(abs(result$Importance), decreasing = TRUE), ]
}

feature_importance <- evaluate_feature_importance(log_model, x_test, y_test)
write.xlsx(feature_importance, "Feature_Importance_Logistic.xlsx", overwrite = TRUE)

