# Heart Disease Prediction Project

## Overview
This project implements a comprehensive machine learning pipeline for predicting heart disease using various patient health indicators. The project includes extensive data preprocessing, exploratory data analysis, feature engineering, and multiple machine learning models with hyperparameter optimization.

## 🎯 Objective
To develop an accurate machine learning model that can predict the presence of heart disease based on patient health metrics, enabling early detection and intervention.

## 📊 Dataset
The project uses two main datasets:
- **`HeartAssign2.csv`**: Primary training dataset containing patient health records
- **`HeartNewPatients.csv`**: Additional patient data for testing/validation

### Features
The dataset includes 13 key health indicators:
- **age**: Age of the patient
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia type (1-3)
- **target**: Heart disease presence (1 = present, 0 = absent)

## 🔧 Technical Implementation

### Data Preprocessing Pipeline
1. **Data Exploration & Quality Assessment**
   - Missing value analysis
   - Duplicate detection
   - Data type validation
   - Statistical summaries

2. **Data Cleaning**
   - Faulty data identification and replacement
   - Missing value imputation using Random Forest
   - Duplicate removal
   - Data type conversion

3. **Feature Engineering**
   - Skewness correction using Yeo-Johnson transformation
   - Standardization using StandardScaler
   - Categorical encoding using LabelEncoder
   - Outlier detection and analysis

4. **Data Analysis**
   - Univariate analysis with histograms and box plots
   - Bivariate analysis with pair plots and correlation matrices
   - Statistical testing (Chi-square, Point-biserial correlation)
   - Multivariate analysis with heatmaps

### Machine Learning Models
The project implements and compares multiple algorithms:

1. **Random Forest**
   - Base model and hyperparameter-tuned version
   - Feature importance analysis
   - Cross-validation optimization

2. **XGBoost**
   - Gradient boosting implementation
   - Advanced hyperparameter tuning
   - Feature importance visualization

3. **K-Nearest Neighbors (KNN)**
   - Distance-based classification
   - Optimal k-value selection
   - Multiple distance metrics evaluation

4. **Support Vector Machine (SVM)**
   - Multiple kernel functions (linear, RBF, polynomial)
   - Regularization parameter optimization
   - Probability estimation enabled

5. **Logistic Regression**
   - Linear classification baseline
   - Regularization techniques (L1, L2, ElasticNet)
   - Statistical interpretation

### Model Evaluation
Each model is evaluated using:
- **Confusion Matrices** for all datasets (train/validation/test)
- **Classification Reports** with precision, recall, F1-score
- **ROC Curves** and AUC scores
- **Feature Importance** analysis (where applicable)
- **Cross-validation** scores

### Hyperparameter Optimization
- **RandomizedSearchCV** for efficient parameter space exploration
- **5-fold cross-validation** for robust model selection
- **AUC-based scoring** for optimal threshold selection

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib scipy joblib
```

### Running the Analysis
1. Clone the repository
2. Ensure datasets are in the `data/` directory
3. Run the Jupyter notebook: `(AM)_ASSIGN1_GOHYICHENG_2206335.ipynb`

### Interactive Prediction
The project includes an interactive prediction system where users can input their health metrics to get real-time heart disease risk assessment:

```python
# The notebook includes a user input function that prompts for:
# - Age, gender, chest pain type
# - Blood pressure, cholesterol levels
# - Heart rate, exercise metrics
# - Other clinical indicators
```

## 📈 Results & Performance
- Comprehensive model comparison across multiple metrics
- Feature importance rankings for interpretability
- Cross-validation scores for model reliability
- ROC curves for threshold optimization

## 🗂️ Project Structure
```
HeartDiseasePrediction/
├── data/
│   ├── HeartAssign2.csv
│   └── HeartNewPatients.csv
├── (AM)_ASSIGN1_GOHYICHENG_2206335.ipynb
├── (AM)_ASSIGN1_GOHYICHENG_2206335.docx
└── README.md
```

## 🔍 Key Features
- **Robust Data Preprocessing**: Handles missing values, outliers, and data quality issues
- **Multiple ML Algorithms**: Comprehensive comparison of different approaches
- **Hyperparameter Optimization**: Automated tuning for optimal performance
- **Interactive Prediction**: Real-time risk assessment capability
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Feature Engineering**: Advanced preprocessing for improved model performance

## 📋 Requirements
- Python 3.7+
- Jupyter Notebook
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- XGBoost
- scipy, joblib

## 👨‍💻 Collaborators
**Goh Yi Cheng**  
**Toh Yong Cheng**
**Anson Yong Wei Sheng**

## 📄 License
This project is for educational purposes as part of an academic assignment.

---

*This project demonstrates a complete machine learning workflow from data preprocessing to model deployment, emphasizing best practices in data science and machine learning methodology.*
