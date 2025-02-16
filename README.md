Parkinson’s Disease Regression Analysis
Overview
This project aims to predict the Unified Parkinson’s Disease Rating Scale (UPDRS) scores using linear regression models based on voice parameters and other features. The dataset is sourced from the UCI Machine Learning Repository.

Project documentation
Dataset
The dataset contains 5,875 voice recordings from 42 patients, recorded over six months.
It includes 22 features, such as age, sex, voice jitter, shimmer, noise-to-harmonics ratio (NHR), harmonics-to-noise ratio (HNR), and recurrence period density entropy (RPDE).
The target variable is total UPDRS, which measures the severity of Parkinson’s disease.
Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/YOUR_GITHUB_USERNAME/parkinson-regression-analysis.git
cd parkinson-regression-analysis
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the Main Script
bash
Copy
Edit
python main.py
Run Jupyter Notebook for EDA
bash
Copy
Edit
jupyter notebook
Run Tests
bash
Copy
Edit
pytest tests/
Modeling Approach
Data Preprocessing

Handle missing values (if any).
Standardize numerical features for better regression performance.
Feature Engineering

Extract relevant features from voice data.
Normalize features to ensure consistency.
Regression Model

Implement Linear Regression to predict total UPDRS.
Evaluate performance using Mean Absolute Error (MAE) and R-squared (R²).
Visualization

Correlation heatmaps to analyze feature relationships.
Scatter plots of predicted vs actual UPDRS values.
Results
The regression model helps estimate UPDRS scores based on voice parameters.
This approach provides an automated, objective method for monitoring Parkinson’s progression.
Future Improvements
Implement more advanced regression models (e.g., Random Forest, Neural Networks).
Incorporate feature selection techniques to improve prediction accuracy.
Deploy as a web-based application for real-time monitoring.
Contributors
👤 Your Name
📧 mahdi.rajaee@ymail.com
🔗 LinkedIn : https://www.linkedin.com/in/mahdi-rajaee-a815a086/

License
This project is licensed under the MIT License.