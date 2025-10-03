# Loan Default Prediction — README
## Project Overview

A machine learning project to predict loan defaults using a 255,347-row loan dataset. The repository contains data exploration, preprocessing, model training, evaluation, and notes on limitations and recommended fixes. The work compares a Logistic Regression baseline with a Random Forest model and documents observed issues that may have inflated the Random Forest’s reported performance. Full project report is included. 

## Quick summary
- Goal: Classify loans as Default = 1 (default) or Default = 0 (non-default).
- Rows: 255,347.
- Main models evaluated: Logistic Regression (baseline) and Random Forest (ensemble).
- Key finding: Logistic Regression shows reasonable separation (Accuracy ≈ 0.68, ROC AUC ≈ 0.745). Random Forest reports very high performance (≈98% accuracy, ROC AUC ≈0.98), but those results are likely inflated by preprocessing/data-leakage issues described in the   report. 

## Dataset & Features

The dataset contains numeric features (e.g., Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio) and categorical features (e.g., Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner) plus the Default target. No missing values were reported in the version used for analysis. 

## Methodology 

- Data cleaning & basic EDA.
- Encode categorical variables (label mapping + one-hot).
- Scale numeric features.
- Address class imbalance (RandomOverSampler used in the notebook).
- Train & evaluate models (Logistic Regression, Random Forest).
- Plot confusion matrices, ROC curves, and classification reports. 
- Important caveat: The notebook performed oversampling and some preprocessing steps in an order that introduced data leakage (oversampling before train/test split and scaler fit on test set). This may have produced overly optimistic metrics for the Random Forest. See the full report for details and recommended pipeline fixes. 

## Results (at-a-glance)
- Logistic Regression: Accuracy ≈ 0.68, ROC AUC ≈ 0.745.
- Random Forest: Accuracy ≈ 0.98, ROC AUC reported ≈ 0.98. \n
* See the report for detailed confusion matrices, classification reports, and a discussion of why the Random Forest results should be revalidated. *
