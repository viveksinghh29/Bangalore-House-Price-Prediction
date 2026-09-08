# Bangalore House Price Prediction

### Python | Pandas | NumPy | Scikit-Learn | Gradient Boosting | Streamlit | Joblib

# Project Overview

**Bangalore House Price Prediction** is an end-to-end Machine Learning project that predicts residential property prices in Bangalore based on location, availability, total square footage, BHK, bathrooms, and balconies.

The project follows the complete Machine Learning lifecycle — from raw dataset cleaning and feature engineering to model comparison, evaluation, pipeline creation, and deployment-ready Streamlit application development.

The application provides users with an interactive interface where they can enter property details and receive an estimated house price in **Indian Lakh/Crore format**.

A major focus of this project is **building a reliable and honest ML pipeline**. During development, a data leakage issue was identified and corrected. The final model excludes all price-derived features and reports its actual test-set performance.

---

# Problem Statement

Real-estate prices in Bangalore vary significantly depending on location, property size, BHK configuration, bathrooms, balconies, and availability.

For buyers, sellers, and real-estate platforms, manually estimating property prices can be difficult and inconsistent.

This project aims to build a Machine Learning system that can provide an estimated property price based on historical Bangalore housing data.

The system answers questions such as:

* How much could a property cost based on its location?
* How does total square footage affect property price?
* How does the number of bedrooms influence price?
* How do bathrooms and balconies contribute to property value?
* Can Machine Learning provide a reliable property price estimate?

---

# Objectives

* Build an end-to-end house price prediction system
* Clean and preprocess real-world housing data
* Perform feature engineering
* Handle missing values and outliers
* Encode categorical variables
* Compare multiple Machine Learning models
* Prevent data leakage
* Build a reproducible Scikit-Learn pipeline
* Evaluate model performance using standard regression metrics
* Develop an interactive Streamlit prediction application
* Provide price predictions in Indian Lakh/Crore notation

---

# Dataset Description

**Dataset:** Bangalore House Price Dataset

### Dataset Size

* Original Listings: **13,320**
* Rows Used After Cleaning & Outlier Removal: **12,530**
* Final Features: **6**

### Features Used

* Location
* Availability
* Total Square Feet
* Bathrooms
* Balconies
* BHK

### Target Variable

* **Price**

The original dataset also contained additional columns such as `society` and `size`. These were processed or removed during feature engineering.

---

# Tools & Technologies

### Programming

* Python 3.12
* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* ColumnTransformer
* OneHotEncoder
* Pipeline
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost

### Application

* Streamlit
* Joblib

### Data Analysis & Visualization

* Matplotlib
* Seaborn
* Jupyter Notebook

### Development Tools

* Git
* GitHub
* VS Code

---

# Project Workflow

* Data Collection
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Outlier Detection & Removal
* Exploratory Data Analysis
* Categorical Encoding
* Model Training
* Model Comparison
* Data Leakage Detection
* Model Retraining
* Model Evaluation
* Pipeline Creation
* Model Serialization
* Streamlit Application Development
* Input Validation
* Price Prediction

---

# Data Preprocessing

The raw housing dataset required several preprocessing steps before Machine Learning could be applied.

### Missing Value Handling

* `society` was removed because of a high percentage of missing values.
* Missing `balcony` values were handled using the mean.
* Missing `bath` values were handled using the median.
* Missing `location` values were handled using a fallback category.
* Missing `size` values were handled using a default/mode-like value.

### Feature Extraction

The `size` column contained values such as:

```text
2 BHK
3 BHK
4 Bedroom
```

The number of bedrooms was extracted using regular expressions and converted into a numerical `bhk` feature.

### Total Square Footage

Some properties contained square footage ranges such as:

```text
2100 - 2850
```

These ranges were converted into a single numerical value by calculating their average.

### Location Processing

Locations with fewer than 10 listings were grouped into an `"other"` category.

This resulted in approximately **254 final location categories**.

### Outlier Handling

Properties with an unrealistic:

```text
total_sqft / bhk
```

ratio below **300 sqft per bedroom** were removed.

BHK values were also capped at **8** to reduce the effect of extreme outliers.

---

# Feature Engineering

The final model uses the following features:

| Feature        | Description                  |
| -------------- | ---------------------------- |
| `location`     | Property location            |
| `availability` | Property availability status |
| `total_sqft`   | Total property area          |
| `bath`         | Number of bathrooms          |
| `balcony`      | Number of balconies          |
| `bhk`          | Number of bedrooms           |

Categorical features such as `location` and `availability` are encoded using **OneHotEncoder**.

Numerical features are passed directly through the preprocessing pipeline.

All preprocessing and model steps are combined into a single **Scikit-Learn Pipeline** to ensure that training and inference use the same transformations.

---

# Exploratory Data Analysis

The exploratory analysis focuses on understanding the relationship between property characteristics and price.

The project analyzes:

* Property price distribution
* Price per square foot
* BHK distribution
* Bathroom distribution
* Balcony distribution
* Location-wise pricing
* Total square footage
* Availability status
* Area type distribution
* Property size vs price
* BHK vs price
* Location vs price

---

# Machine Learning

Multiple regression algorithms were evaluated to determine the most suitable model for the project.

### Models Compared

* Random Forest — Baseline
* Random Forest — Tuned
* Gradient Boosting Regressor
* XGBoost

---

# Model Evaluation

The models were evaluated using:

* RMSE
* MAE
* R² Score

### Model Comparison

| Model                         |      RMSE |       MAE |   R² Test |  R² Train |
| ----------------------------- | --------: | --------: | --------: | --------: |
| Random Forest — Baseline      |     91.07 |     31.71 |     0.612 |     0.927 |
| Random Forest — Tuned         |     89.98 |     33.75 |     0.621 |     0.804 |
| **Gradient Boosting — Final** | **87.30** | **34.34** | **0.643** | **0.794** |
| XGBoost                       |     94.82 |     33.76 |     0.579 |     0.874 |

---

### Gradient Boosting Regressor

```python
GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

The Gradient Boosting model was selected because it achieved the best test-set R² and RMSE among the evaluated models while maintaining a smaller train/test performance gap compared with the baseline Random Forest.

### Final Test Performance

* **R² Score:** 0.643
* **MAE:** ≈ 34.3 Lakhs
* **RMSE:** ≈ 87.3 Lakhs
* **Training Rows:** 12,530
* **Features:** 6

---

# Data Leakage Detection & Correction

One of the most important lessons from this project was identifying a **data leakage issue** during development.

An early version of the training pipeline contained:

```text
log_price_per_sqft
```

This feature was calculated using the target variable:

```text
price
```

Because the actual price would not be available when a user makes a prediction, this feature represented information that would not exist at inference time.

The leaked feature caused the initial model to report an inflated:

```text
R² = 0.973
```

This was not genuine predictive performance.

The pipeline was therefore rebuilt from the raw dataset with the price-derived feature completely removed.

The final model reports:

```text
R² = 0.643
```

This lower score represents the model's **honest performance without target leakage**.

The leakage correction and complete retraining process are documented in:

```text
training/retrain_model.ipynb
```

---

# Streamlit Application

The trained model is integrated into an interactive Streamlit web application.

Users can enter:

* Location
* Availability
* Total Square Feet
* BHK
* Bathrooms
* Balconies

The application then processes the inputs through the saved ML pipeline and generates a predicted property price.

### Application Features

* Interactive prediction interface
* Location selection
* Property configuration inputs
* Input validation
* Out-of-distribution warnings
* Indian Lakh/Crore price formatting
* Persistent prediction results
* Responsive Streamlit interface
* Ocean Blue visual design
* "About the Model" section
* Real evaluation metrics displayed in the UI

---

# Application Architecture


```text
User
  ↓
Streamlit Interface
  ↓
Input Validation
  ↓
Feature Construction
  ↓
Saved Scikit-Learn Pipeline
  ↓
OneHotEncoder + Preprocessing
  ↓
Gradient Boosting Regressor
  ↓
Predicted Property Price
  ↓
Indian Lakh/Crore Formatting
```

---

# Model Artifacts


The trained model and supporting metadata are stored inside the `model/` directory.

### Files

```text
Bangalore-project.joblib
feature_columns.json
categories.json
metrics.json
```

### Bangalore-project.joblib

Contains the complete Scikit-Learn pipeline, including preprocessing and the trained Gradient Boosting model.

### feature_columns.json

Stores the exact feature order expected by the model.

### categories.json

Stores valid categorical values used by the application.

### metrics.json

Stores the model's evaluation metrics displayed in the application.

---

# Input Validation

The application validates user inputs before making predictions.

Validation includes:

* Valid BHK range
* Valid bathroom count
* Valid balcony count
* Valid total square footage
* Valid location categories
* Valid availability categories
* Detection of potentially unusual property configurations

The application uses contextual warnings for unusual inputs instead of unnecessarily blocking users from making predictions.

---

# Project Structure

```text
Bangalore-House-Price-Prediction/
│
├── app.py
│
├── model/
│   ├── Bangalore-project.joblib
│   ├── feature_columns.json
│   ├── categories.json
│   └── metrics.json
│
├── assets/
│   ├── style.css
│   └── bangalore.jpg
│
├── utils/
│   ├── __init__.py
│   ├── theme.py
│   ├── hero.py
│   ├── inputs.py
│   ├── prediction.py
│   ├── result.py
│   ├── validation.py
│   └── insights.py
│
├── training/
│   ├── retrain_model.ipynb
│   └── BHP.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---


### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# Model Retraining

The complete model retraining process is available in:

```text
training/retrain_model.ipynb
```

The notebook includes:

* Data loading
* Data cleaning
* Missing value handling
* Feature engineering
* Outlier removal
* Data leakage correction
* Train/test split
* Model training
* Model comparison
* Model evaluation
* Final model training
* Artifact export

Training dependencies can be installed using:

```bash
pip install -r requirements.txt -r training/requirements-training.txt
```

The notebook exports the updated model artifacts into:

```text
training/model_out/
```

These files can then be copied into:

```text
model/
```

to update the application.

### Live Application

**Live App:** Add deployment link after deployment.

---

# Key Insights

* Location is an important factor in Bangalore property pricing.
* Property size has a strong relationship with overall price.
* BHK configuration influences property valuation.
* Bathrooms and balconies provide additional information about property characteristics.
* Extremely small area-per-bedroom ratios can create unrealistic training examples.
* Proper preprocessing is critical when dealing with real-world housing data.
* Tree-based ensemble models can capture nonlinear relationships between property characteristics and price.
* Preventing target leakage is essential for obtaining trustworthy Machine Learning metrics.

---

# Business Impact


This project demonstrates how Machine Learning can support real-estate decision-making by:

* Providing quick property price estimates
* Supporting property valuation
* Helping buyers compare potential properties
* Helping sellers estimate market prices
* Demonstrating location-based price patterns
* Reducing reliance on manual price estimation
* Providing a foundation for real-estate prediction systems

---

# Challenges Faced

* Cleaning real-world housing data
* Handling missing values
* Converting text-based property sizes into numerical features
* Processing square-footage ranges
* Managing high-cardinality location data
* Detecting and removing unrealistic outliers
* Comparing multiple regression algorithms
* Identifying target leakage
* Rebuilding the pipeline after leakage correction
* Maintaining consistency between training and inference
* Designing an interactive prediction interface

---

# Key Learnings

[svg](https://github.com/viveksinghh29/Bangalore-House-Price-Prediction#key-learnings)

* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Outlier Detection
* Categorical Encoding
* Regression Models
* Ensemble Learning
* Model Evaluation
* Data Leakage Prevention
* Scikit-Learn Pipelines
* Model Serialization
* Streamlit Application Development
* Input Validation
* ML Model Deployment

---

# Future Improvements

[svg](https://github.com/viveksinghh29/Bangalore-House-Price-Prediction#future-improvements)

* Improve model performance with additional location and property features
* Experiment with advanced boosting algorithms
* Perform systematic hyperparameter optimization
* Add confidence or prediction intervals
* Add interactive price analysis by location
* Add visual EDA dashboards
* Add model explainability using SHAP
* Deploy the application publicly
* Add automated model retraining pipelines

---

# Author

**Vivek Kumar Singh**

AI & Machine Learning | Data Science | Software Development

---

⭐ **If you found this project useful, consider giving it a star!**
