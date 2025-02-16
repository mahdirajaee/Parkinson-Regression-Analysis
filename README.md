# **Parkinson’s Disease Regression Analysis**  

## 🚀 **Project Overview**  
This project implements **linear regression** to predict the **Unified Parkinson’s Disease Rating Scale (UPDRS)** based on **voice parameters** and other patient features. The goal is to develop an **automated system** that helps neurologists optimize treatments by continuously monitoring disease progression.  

## 📊 **Dataset**  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)  
- **Records:** 5,875 voice samples collected from **42 patients** over six months  
- **Features:**  
  - **Demographic:** Age, sex  
  - **Voice Parameters:** Jitter, shimmer, noise-to-harmonics ratio (NHR), harmonics-to-noise ratio (HNR)  
  - **Nonlinear Measures:** Recurrence Period Density Entropy (RPDE), Detrended Fluctuation Analysis (DFA), Perceived Vocal Effort (PPE)  
- **Target Variable:** **Total UPDRS** (measures Parkinson’s severity)  

## ⚙️ **Installation**  

### **Clone the Repository**  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/parkinson-regression-analysis.git
cd parkinson-regression-analysis
```

### **Install Dependencies**  
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ **How to Run the Project**  

### **Run the Main Script**  
```bash
python main.py
```

### **Run Jupyter Notebook for Exploratory Data Analysis (EDA)**  
```bash
jupyter notebook
```

### **Run Tests**  
```bash
pytest tests/
```

## 🔍 **Modeling Approach**  

1. **Data Preprocessing**  
   - Handle missing values  
   - Normalize and scale features  
2. **Feature Engineering**  
   - Extract and select relevant voice-based predictors  
3. **Regression Model**  
   - Train a **Linear Regression** model to predict **total UPDRS**  
   - Evaluate performance using **Mean Absolute Error (MAE)** and **R-squared (R²)**  
4. **Visualization**  
   - Correlation heatmaps for feature relationships  
   - Scatter plots for predicted vs actual UPDRS values  

## 📈 **Results & Insights**  
- The model provides an **objective, automated** method for monitoring Parkinson’s progression.  
- The approach allows **continuous evaluation** of patients using **simple voice recordings**.  

## 🚀 **Future Improvements**  
- Implement **advanced regression models** (e.g., **Random Forest, Neural Networks**)  
- Incorporate **feature selection techniques** to improve accuracy  
- Develop a **real-time monitoring application**  

## 👤 **Author**  
📧 **Mahdi Rajaee**  
🔗 **https://www.linkedin.com/in/mahdi-rajaee-a815a086/**  

## 📜 **License**  
This project is licensed under the **MIT License**.  
