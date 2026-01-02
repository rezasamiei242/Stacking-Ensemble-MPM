"""
Support Vector Regression (SVR) base learner
--------------------------------------------
This script trains an SVR model, performs hyperparameter optimization,
evaluates predictive performance, and generates prospectivity predictions
for the target area.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
# 2. Hyperparameter optimization using Random Search
# --------------------------------------------------
param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.01, 0.1, 10, 100, 1000],
    "gamma": ["scale", "auto"],
}

svr = SVR()

random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="r2",
    random_state=42,
    verbose=1,
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("Best hyperparameters:", random_search.best_params_)


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
f1_data["SVR_Prediction"] = f1_predictions

f1_data.to_excel("outputs/F1_SVR_predictions.xlsx", index=False)


# --------------------------------------------------
# 5. Spatial visualization
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(f1_data["X"], f1_data["Y"], c=f1_predictions, cmap="viridis")
plt.colorbar(label="SVR prediction")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("SVR-based prospectivity map")
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

