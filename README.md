[Wersja polska (Polish version)](./README_PL.md)
# survey-analysis-ml
Predictive modeling and analysis of mental health in the tech industry using Logistic Regression, SVM, and MLP.

The dataset contains survey responses from technology industry employees regarding workplace environment, mental health support, benefits, and treatment history.

---
## Dataset
The dataset used in this project is the Mental Health in Tech Survey from 2014, available publicly on Kaggle:
https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

---

## Project Background

This project was originally developed as part of a university coursework assignment focused on machine learning model evaluation and comparison.

The original academic project included additional classification models and experiments. For the purpose of this repository, the project was refined and simplified to present the three final models selected for comparison:

* Logistic Regression with RFECV
* Support Vector Machine (SVM)
* Multilayer Perceptron (MLP)

The objective was to compare different classification approaches on structured survey data and evaluate their performance using cross-validation and a separate holdout test set.

---

## Project Structure & Workflow
### 1. Data Cleaning & Feature Engineering
* Handled missing values using domain-informed assumptions (e.g., `self_employed`, `work_interfere`)
* Standardized and simplified noisy categorical values in the `Gender` column into three classes: `male`, `female`, and `other`.
* Removed invalid and extreme values from the `Age` column to ensure data quality.
* Encoded categorical variables using:
  * Ordinal encoding for structured variables with inherent order (e.g., `work_interfere`, `leave`, `no_employees`)
  * One-hot encoding for nominal features using `pd.get_dummies`
* Scaled numerical features using `MinMaxScaler` fitted only on the training set to prevent data leakage.

### 2. Exploratory Data Analysis (EDA)
* Analyzed relationships between key workplace and demographic factors and mental health treatment using grouped count plots.
* Investigated how organizational support features (e.g., anonymity, supervisor support, benefits) relate to treatment-seeking behavior.

### 3. Feature Selection (RFECV)
* Applied Recursive Feature Elimination with Cross-Validation (RFECV) using Logistic Regression as the base estimator.
* Optimized feature set based on F1-score using 5-fold cross-validation.
* Reduced dimensionality by removing weak or redundant features.

### 4. Model Architecture & Cross-Validation
Three models were trained and compared:
* Logistic Regression (optimized features): Interpretable baseline model trained on the RFECV-selected feature subset.
* Support Vector Machine (SVM, RBF kernel): Non-linear model used to capture complex relationships in the data.
* Multilayer Perceptron (MLP): Neural network with two hidden layers (10, 5), using tanh activation and the adam optimizer to model non-linear patterns.

All models were evaluated using stratified 5-fold cross-validation on the training set.

### 5. Evaluation and Performance Benchmarking
* Final evaluation was performed on a separate holdout test set.
* Metrics used: Accuracy, Precision, Recall, F1-score.
* Confusion matrices were used to analyze classification errors.

---

## Performance Summary

Final evaluation metrics collected on the holdout test set demonstrated the following results:

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (RFECV)** | 0.797 | 0.797 | 0.803 | 0.800 |
| **SVM (RBF Kernel)** | 0.825 | 0.817 | 0.843 | 0.829 |
| **Multilayer Perceptron (MLP)** | **0.845** | **0.828** | **0.874** | **0.851** |

### Key Takeaways:
1. All models achieved relatively strong performance on structured survey data.
2. The MLP model achieved the highest overall scores, though differences between models were moderate.
3. SVM provided a strong balance between performance and generalization.
4. Logistic Regression offered the best interpretability with competitive baseline results.

---

## Limitations

* The dataset contains approximately 1,200 survey responses, which is relatively small for training neural network models.
* The MLP model was included primarily for educational and comparative purposes as part of the model evaluation process.
* Results should be interpreted as a comparison of machine learning approaches on this dataset rather than as a production-ready predictive solution.

---

## Tech Stack & Dependencies
* **Core Language:** Python 3.x
* **Exploratory Analytics:** `pandas`, `numpy`
* **Visualization Suite:** `matplotlib`, `seaborn`
* **Machine Learning Engine:** `scikit-learn`
