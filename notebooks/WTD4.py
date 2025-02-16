def run_experiment(seed):
    np.random.seed(seed)
    
    # Shuffle and split dataset
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df_shuffled.iloc[:split_index].reset_index(drop=True)
    test_df = df_shuffled.iloc[split_index:].reset_index(drop=True)

    # Normalize dataset
    X_train = scaler.fit_transform(train_df.drop(columns=["total_UPDRS"]))
    y_train = train_df["total_UPDRS"].values
    X_test = scaler.transform(test_df.drop(columns=["total_UPDRS"]))
    y_test = test_df["total_UPDRS"].values

    # Train models
    y_pred_LLS = X_test @ np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
    y_pred_local = local_regression(X_train, y_train, X_test, N=10)

    # Evaluate
    return evaluate(y_test, y_pred_LLS), evaluate(y_test, y_pred_local)

# Run 20 experiments with different seeds
seeds = np.random.randint(1, 999999, size=20)
results_LLS = []
results_Local = []

for s in seeds:
    res_LLS, res_Local = run_experiment(s)
    results_LLS.append(res_LLS)
    results_Local.append(res_Local)

# Compute averages
avg_LLS = np.mean(results_LLS, axis=0)
avg_Local = np.mean(results_Local, axis=0)

# Display results
final_df = pd.DataFrame({
    "Metric": ["MSE", "R²", "Correlation"],
    "Average LLS Regression": avg_LLS,
    "Average Local Regression (N=10)": avg_Local
})
print(final_df)
