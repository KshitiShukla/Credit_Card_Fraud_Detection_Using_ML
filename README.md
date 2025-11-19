# Credit_Card_Fraud_Detection_Using_ML

## Project Overview

This project is a classic supervised machine learning application focused on detecting fraudulent credit card transactions. Given the highly imbalanced nature of financial datasets (where non-fraudulent transactions vastly outnumber fraudulent ones), the primary challenge is building a model that can accurately identify the rare, critical events of fraud while maintaining low false positives.

The solution involves a comprehensive data science workflow, from data exploration and handling class imbalance to training and evaluating various classification models.

## Important Note on the Dataset

Due to GitHub's limitations regarding large file sizes, the original CSV database file containing the transaction data has not been uploaded to this repository.

Data Source: You can find the dataset, typically containing anonymized transactional features (V1 to V28), the Time, Amount, and the binary Class label, at the following source:

Database Link: https://www.kaggle.com/code/priyang/credit-card-fraud-detect-under-over-sampling/input?select=creditcard.csv

Please download the file from the link above and place it in the project root directory before running the notebook.

## Project Workflow: How It Works

The project follows a standard machine learning pipeline structured to address the specific challenges of fraud detection:

1. Data Collection & Initial Setup

Importing Libraries: Load all necessary Python libraries (pandas, numpy, sklearn, etc.).

Data Loading: Read the creditcard.csv file into a pandas DataFrame.

2. Exploratory Data Analysis (EDA)

Data Inspection: Examine the first few rows, check data types, and look for missing values.

Class Imbalance Analysis: Crucially, visualize and quantify the severe imbalance between the 'Non-Fraud' (Class 0) and 'Fraud' (Class 1) transactions. This step dictates the rest of the modeling approach.

3. Data Preprocessing & Feature Engineering

Scaling: Scale the Time and Amount features to standardize the input, as the other features (V1-V28) are already the result of a PCA transformation.

Splitting Data: Divide the dataset into training and testing sets.

4. Handling Class Imbalance

Since the dataset is highly skewed, techniques are employed to balance the training data, allowing models to learn the patterns of the minority (fraudulent) class effectively. This may include:

Oversampling (e.g., SMOTE): Creating synthetic samples of the minority class.

Undersampling: Reducing the number of samples in the majority class.

5. Model Training & Selection

Train multiple classification models to find the best performer on this highly skewed data. Common models include:

Logistic Regression (Baseline)

Decision Tree Classifier

Random Forest Classifier

XGBoost / Gradient Boosting (Advanced)

6. Evaluation & Results

Evaluate the selected models on the untouched test dataset.

Key Metrics: Given the nature of fraud detection, performance is evaluated using metrics like Precision, Recall, F1-Score, and the Confusion Matrix, as simple Accuracy can be misleading.

## Technology Stack & Libraries

Python: The core programming language.

pandas and numpy: Data manipulation and numerical operations.

matplotlib and seaborn: Data visualization for EDA and results presentation.

scikit-learn (sklearn): Comprehensive library for machine learning, including:

train_test_split, StandardScaler (for preprocessing).

Model implementations (LogisticRegression, RandomForestClassifier, etc.).

Evaluation metrics (confusion_matrix, classification_report).

Imbalance Handling Libraries (e.g., imblearn): Specifically for techniques like SMOTE.
