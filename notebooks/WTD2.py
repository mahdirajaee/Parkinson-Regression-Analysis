# from scipy.spatial.distance import cdist

# def local_regression(X_train, y_train, X_test, N=10, lr=0.001, tol=1e-6, max_iter=1000):
#     y_pred = np.zeros(X_test.shape[0])
    
#     for i in range(X_test.shape[0]):
#         # Find the N nearest neighbors
#         distances = cdist(X_train, X_test[i].reshape(1, -1), metric="euclidean").flatten()
#         nearest_idxs = np.argsort(distances)[:N]
#         X_local = X_train[nearest_idxs]
#         y_local = y_train[nearest_idxs]

#         # Train using steepest descent
#         w_local = steepest_descent(X_local, y_local, lr, tol, max_iter)
#         y_pred[i] = X_test[i] @ w_local
    
#     return y_pred

# y_pred_local = local_regression(X_train, y_train, X_test, N=10)

# # Print performance metrics
# print("Local Regression (N=10) Results:", evaluate(y_test, y_pred_local))
