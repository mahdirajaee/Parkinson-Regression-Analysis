# def plot_regression(y_true, y_pred, title="Predicted vs True UPDRS"):
#     plt.figure(figsize=(6, 4))
#     plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
#     min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
#     plt.xlabel("True UPDRS")
#     plt.ylabel("Predicted UPDRS")
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.show()

# plot_regression(y_test, y_pred_LLS, "LLS Regression: True vs Predicted")
# plot_regression(y_test, y_pred_local, "Local Regression: True vs Predicted")

# # Plot error histograms
# def plot_error_histogram(errors, title="Error Histogram"):
#     plt.figure(figsize=(6, 4))
#     plt.hist(errors, bins=50, density=True, histtype='bar')
#     plt.xlabel("Error (y - ŷ)")
#     plt.ylabel("Density")
#     plt.title(title)
#     plt.grid()
#     plt.show()

# plot_error_histogram(y_test - y_pred_LLS, "LLS Regression Error Histogram")
# plot_error_histogram(y_test - y_pred_local, "Local Regression Error Histogram")

# # Generate performance comparison table
# performance_df = pd.DataFrame({
#     "Metric": ["MSE", "R²", "Correlation"],
#     "LLS Regression": evaluate(y_test, y_pred_LLS),
#     "Local Regression (N=10)": evaluate(y_test, y_pred_local)
# })
# print(performance_df)
