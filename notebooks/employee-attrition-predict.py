import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5 # For cross-validation
DATA_DIR = 'data'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Initial Exploration ---
print("--- 1. Data Loading and Initial Exploration ---")

def load_and_explore_data(file_path):
    """
    Loads the dataset, performs initial exploration, and returns the DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=';', header=None, skiprows=1)

        column_names = [
            'Employee ID', 'Age', 'Gender', 'Years at Company', 'Job Role', 'Monthly Income',
            'Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Number of Promotions',
            'Overtime', 'Distance from Home', 'Education Level', 'Marital Status',
            'Number of Dependents', 'Job Level', 'Company Size', 'Company Tenure',
            'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities',
            'Company Reputation', 'Employee Recognition', 'Attrition'
        ]
        df.columns = column_names

        print(f"Data loaded successfully. Dimensions: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData info:")
        print(df.info())
        print("\nDescriptive statistics:")
        print(df.describe(include='all'))

        return df

    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Ensure the file is in the correct directory.")
        exit()

# Load the training and testing datasets
train_file = os.path.join(DATA_DIR, 'train.csv')
test_file = os.path.join(DATA_DIR, 'test.csv')

df_train = load_and_explore_data(train_file)
df_test = load_and_explore_data(test_file)

# Combine train and test for preprocessing (optional, but good for consistent encoding)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- 2. Data Preprocessing and Feature Engineering ---")

def preprocess_data(df):
    """
    Preprocesses the data, handles missing values, encodes categorical features,
    and performs feature engineering.
    """
    # Convert 'Attrition' to binary (0/1)
    df['Attrition'] = df['Attrition'].map({'Stayed': 0, 'Left': 1})
    print(f"Attrition value counts after mapping: {df['Attrition'].value_counts()}")

    # Identify column types
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove identifiers and target from feature lists
    for col in ['Employee ID', 'Attrition']:
        if col in numerical_cols:
            numerical_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

    print(f"\nNumerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")

    # Ordinal mapping
    ordinal_mapping = {
        'Work-Life Balance': ['Poor', 'Fair', 'Good', 'High', 'Excellent'],
        'Job Satisfaction': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'Performance Rating': ['Low', 'Below Average', 'Average', 'High', 'Very High'],
        'Company Reputation': ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'],
        'Employee Recognition': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    }

    for col, order in ordinal_mapping.items():
        if col in df.columns and col in categorical_cols:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
            df[col] = df[col].cat.codes
            numerical_cols.append(col)
            categorical_cols.remove(col)
        else:
            print(f"Warning: Ordinal column '{col}' not found or already processed.")

    # Feature Engineering
    df['Income_Per_Year_at_Company'] = df['Monthly Income'] / (df['Years at Company'].replace(0, 0.1))
    numerical_cols.append('Income_Per_Year_at_Company')

    # Imputation
    print("\nMissing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    return df, numerical_cols, categorical_cols

df, numerical_cols, categorical_cols = preprocess_data(df)

# --- 3. Data Splitting and Preprocessing Pipelines ---
print("\n--- 3. Data Splitting and Preprocessing Pipelines ---")

# Separate features and target variable
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

# --- 4. Handling Imbalanced Data ---
print("\n--- 4. Handling Imbalanced Data ---")

# Apply SMOTE to the training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"Shape of X_train_resampled after SMOTE: {X_train_resampled.shape}")
print(f"Shape of y_train_resampled after SMOTE: {y_train_resampled.shape}")

# --- 5. Model Training and Evaluation ---
print("\n--- 5. Model Training and Evaluation ---")

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv=5):
    """
    Trains, evaluates, and saves the model. Performs cross-validation and generates reports.
    """
    print(f"\nTraining {model_name}...")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {cv_scores}")
    print(f"Mean cross-validation ROC AUC score: {cv_scores.mean():.4f}")

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save metrics to a dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": roc_auc
    }

    # Save the model
    model_path = os.path.join(MODEL_DIR, f'{model_name.replace(" ", "_")}_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n{model_name} model saved to {model_path}")

    return model, metrics, cm, fpr, tpr, roc_auc

# Initialize models
log_reg_model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced')
rf_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')

# Train and evaluate Logistic Regression
log_reg_model, log_reg_metrics, log_reg_cm, log_reg_fpr, log_reg_tpr, log_reg_auc = train_and_evaluate_model(
    log_reg_model, X_train_resampled, y_train_resampled, X_test_processed, y_test, "Logistic Regression", cv=N_SPLITS
)

# Train and evaluate Random Forest
rf_model, rf_metrics, rf_cm, rf_fpr, rf_tpr, rf_auc = train_and_evaluate_model(
    rf_model, X_train_resampled, y_train_resampled, X_test_processed, y_test, "Random Forest", cv=N_SPLITS
)

# --- 6. Visualization and Reporting ---
print("\n--- 6. Visualization and Reporting ---")

def generate_plots(cm_lr, cm_rf, fpr_lr, tpr_lr, roc_auc_lr, fpr_rf, tpr_rf, roc_auc_rf):
    """
    Generates and saves confusion matrix and ROC curve plots.
    """
    # Confusion Matrix - Logistic Regression
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
    plt.title('Confusion Matrix: Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix_lr.png'))
    plt.close()

    # Confusion Matrix - Random Forest
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
    plt.title('Confusion Matrix: Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix_rf.png'))
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    plt.close()

# Generate plots
generate_plots(log_reg_cm, rf_cm, log_reg_fpr, log_reg_tpr, log_reg_auc, rf_fpr, rf_tpr, rf_auc)

print("Plots generated and saved to the 'reports' directory.")

# --- 7. Conclusion ---
print("\n--- 7. Conclusion ---")
print("Employee attrition prediction models trained and evaluated successfully.")
print("Results, models, and visualizations are saved for further analysis and reporting.")
