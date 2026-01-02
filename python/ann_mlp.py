"""
Artificial Neural Network (ANN-MLP) base learner
------------------------------------------------
This script trains an MLP regressor, performs hyperparameter tuning,
evaluates predictive performance, and generates prospectivity predictions
for the target area.

Author: [MohammadReza Samiei]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load training and test datasets
# --------------------------------------------------
train_df = pd.read_excel("data/Train.xlsx")
test_df = pd.read_excel("data/Test.xlsx")

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]


# --------------------------------------------------
# 2. Hyperparameter tuning using Grid Search
# --------------------------------------------------
mlp = MLPRegressor(max_iter=500, random_state=42)

param_grid = {
    "hidden_layer_sizes": [(50, 50), (100, 100)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
}

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    verbose=1,
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Best hyperparameters:", grid_search.best_params_)


# --------------------------------------------------
# 3. Model evaluation on test data
# --------------------------------------------------
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r = np.sqrt(r2)

print("\nEvaluation metrics:")
print(f"R: {r:.3f}")
print(f"R²: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")


# --------------------------------------------------
# 4. Prediction for target area (F1)
# --------------------------------------------------
f1_data = pd.read_excel("data/F1.xlsx")
X_f1 = f1_data.drop(columns=["X", "Y"])

f1_predictions = best_model.predict(X_f1)
f1_data["ANN_Prediction"] = f1_predictions

f1_data.to_excel("outputs/F1_ANN_predictions.xlsx", index=False)


# --------------------------------------------------
# 5. Spatial visualization
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(f1_data["X"], f1_data["Y"], c=f1_predictions, cmap="viridis")
plt.colorbar(label="ANN prediction")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("ANN-based prospectivity map")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Permutation-based feature impact analysis
# --------------------------------------------------
def permutation_impact(model, features):
    baseline_pred = model.predict(features)
    baseline_mse = mean_squared_error(baseline_pred, baseline_pred)

    impacts = []
    for col in features.columns:
        shuffled = features.copy()
        shuffled[col] = np.random.permutation(shuffled[col])
        shuffled_pred = model.predict(shuffled)
        mse_shuffled = mean_squared_error(baseline_pred, shuffled_pred)
        impacts.append((col, mse_shuffled - baseline_mse))

    return sorted(impacts, key=lambda x: abs(x[1]), reverse=True)


feature_impacts = permutation_impact(best_model, X_f1)
impact_df = pd.DataFrame(feature_impacts, columns=["Feature", "Impact"])
print(impact_df)

