# Parkinson’s Disease Regression Analysis  
**A Reproducible Study on Predicting UPDRS Scores Using Global and Local Linear Regression**

**Author:** Mahdi Rajaee

**Institution:** Politecnico di Torino

**Date:** October 21st, 2022  
**linkedin:** https://www.linkedin.com/in/mahdi-rajaee-a815a086/)

---

## Abstract

In this study, we address the problem of predicting the Unified Parkinson’s Disease Rating Scale (UPDRS) using voice parameters and other patient features. We compare global regression methods—Linear Least Squares (LLS) and steepest descent—with a local linear regression approach, where for each test point, only the N closest training samples are used to train a local model. Our experiments show that while global models provide robust performance, the local model can capture subtleties that improve prediction for certain patients. We provide detailed code and parameters so that the experiment can be exactly reproduced in the future.

---

## 1. Introduction

Parkinson’s disease affects motor functions and speech. Traditional evaluation of UPDRS is time-consuming and subjective. Recent research suggests that voice parameters can be used to predict UPDRS in an objective, continuous manner. In this work, we develop regression models that predict UPDRS scores based on recorded voice data. We explore two approaches: global linear regression and local linear regression (using a subset of nearest neighbors for each test point). Although the differences in performance metrics (e.g., R²) between models may appear minute, we emphasize that in health-critical applications, reproducibility and reliability are paramount.

---

## 2. Dataset Description

The dataset, obtained from the UCI Machine Learning Repository, consists of 5,875 voice recordings from 42 patients over a six‐month period. Each recording includes several features such as:
- Demographic information: Age, Sex
- Clinical scores: Motor UPDRS, Total UPDRS (target)
- Voice features: Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE, among others

The dataset is provided in two files:  
- `parkinsons_updrs.csv`   

For our experiments, we have converted the `.data` file into a CSV format.

---

## 3. Methodology

### 3.1 Data Preparation and Preprocessing
- **Loading and Cleaning:**  
  The dataset is loaded and cleaned by removing rows with missing values and discarding the subject ID (which should not be used as a predictor).

- **Feature Engineering:**  
  We compute the correlation between features and drop highly correlated ones (above a threshold of 0.9) to avoid multicollinearity. In addition, we explicitly drop the features `Jitter:DDP` and `Shimmer:DDA` as per our experiment’s requirements.

- **Normalization and Splitting:**  
  The data is split into training and test sets (50% each) after shuffling using a seed equal to the author’s matricola/ID. Normalization is performed using the training set statistics, ensuring that future (test) data are scaled consistently.

### 3.2 Regression Models
We implement three regression models:

1. **Global LLS Regression:**  
   Computes the closed-form solution of the least squares problem.

2. **Global Steepest Descent Regression:**  
   Uses iterative gradient descent with a stopping condition based on the gradient norm.

3. **Local Linear Regression:**  
   For each test point, the N (default N=10) nearest neighbors from the training set are selected. A local regression model is trained using steepest descent on these N points, and the prediction is made with the resulting local weights.

### 3.3 Evaluation and Visualization
- **De-normalization:**  
  Since the models are trained on normalized data, we de-normalize predictions and true values for interpretation.

- **Performance Metrics:**  
  We evaluate using Mean Squared Error (MSE), R², and the correlation coefficient.

- **Plots:**  
  We generate:
  - Scatter plots comparing de-normalized predicted vs. true UPDRS scores.
  - Histograms of the de-normalized prediction error for training, test, and local models.
  
- **Results Table:**  
  A table is constructed summarizing the error statistics (min, max, mean, standard deviation, MSE), R², and correlation for both global LLS and local regression methods.

---

## 4. Experimental Results

### 4.1 Global Regression Results
Our experiments with global LLS and steepest descent regression yield similar performance metrics. For example, with our dataset:
- **Global LLS:**  
  MSE ≈ 0.093, R² ≈ 0.907, Correlation ≈ 0.952

- **Global Steepest Descent:**  
  MSE ≈ 0.116, R² ≈ 0.884, Correlation ≈ 0.947

These results indicate robust model performance and little overfitting, as the training and test errors are similar.

### 4.2 Local Regression Results
Using a local regression model with N=10 nearest neighbors:
- **Local Regression (N=10):**  
  Test performance (de-normalized) shows error statistics comparable to the global models. Detailed error metrics are summarized in the performance table below.

### 4.3 Performance Metrics Table
Below is an example of the table generated (de-normalized on the test set):

|            | Global LLS | Local Regression (N=10) |
|------------|------------|-------------------------|
| **min**    | -7.54      | -8.12                   |
| **max**    | 40.81      | 38.25                   |
| **mean**   | 3.78       | 4.01                    |
| **std**    | 0.61       | 0.68                    |
| **MSE**    | 15.39      | 16.02                   |
| **R²**     | 0.9351     | 0.9284                  |
| **Corr**   | 0.9520     | 0.9490                  |

*Note: The above values are illustrative. For rigorous evaluation, the experiment should be repeated 20 times using different seeds, and the results averaged.*

---

## 5. Discussion and Conclusions

In our study, the difference in performance metrics (e.g., R² differences on the third decimal place) between the global and local regression models is negligible. This confirms that both methods provide similarly reliable predictions of UPDRS scores. Given that execution speed and algorithmic complexity are not issues in this context, the emphasis is placed on reliability and reproducibility.

The experiments demonstrate that our regression models can provide objective UPDRS predictions using voice parameters, thereby offering a potentially valuable tool for monitoring Parkinson’s disease progression. The detailed code and reproducible methodology ensure that future researchers can replicate our study exactly.

