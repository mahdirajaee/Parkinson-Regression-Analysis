import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/parkinsons_updrs.csv")

# Drop unnecessary features
df = df.drop(columns=["Jitter:DDP", "Shimmer:DDA"])

# Set seed (replace 123456 with your actual PoliTO ID)
SEED = 123456
np.random.seed(SEED)

# Shuffle and split dataset (50% train, 50% test)
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
N = len(df_shuffled)
split_index = int(N * 0.5)
train_df = df_shuffled.iloc[:split_index].reset_index(drop=True)
test_df = df_shuffled.iloc[split_index:].reset_index(drop=True)

# Normalize dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop(columns=["total_UPDRS"]))
y_train = train_df["total_UPDRS"].values
X_test = scaler.transform(test_df.drop(columns=["total_UPDRS"]))
y_test = test_df["total_UPDRS"].values

# LLS Regression (Closed-form solution)
w_LLS = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
y_pred_LLS = X_test @ w_LLS

# Steepest Descent Regression
def steepest_descent(X, y, lr=0.001, tol=1e-6, max_iter=1000):
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = -2 * (X.T @ (y - X @ w)) / len(y)
        w_new = w - lr * gradient
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w

w_SGD = steepest_descent(X_train, y_train)
y_pred_SGD = X_test @ w_SGD

# Print performance metrics
def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - mse / np.var(y_true)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return mse, r2, corr

print("LLS Regression Results:", evaluate(y_test, y_pred_LLS))
print("Steepest Descent Regression Results:", evaluate(y_test, y_pred_SGD))
