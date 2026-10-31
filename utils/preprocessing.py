from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.runtime import PROJECT_ROOT
TARGET_COL = 'loan_paid_back'
FEATURE_ORDER = [
    'age', 'gender', 'marital_status', 'education_level', 'annual_income',
    'monthly_income', 'employment_status', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'loan_purpose', 'interest_rate', 'loan_term', 'installment',
    'grade_subgrade', 'num_of_open_accounts', 'total_credit_limit', 'current_balance',
    'delinquency_history', 'public_records', 'num_of_delinquencies'
]
CATEGORICAL_COLS = [
    'gender', 'marital_status', 'education_level',
    'employment_status', 'loan_purpose', 'grade_subgrade'
]
NUMERICAL_COLS = [column for column in FEATURE_ORDER if column not in CATEGORICAL_COLS]


def load_data(filepath: str | Path = PROJECT_ROOT / 'dataset' / 'original_dataset.csv'):
    return pd.read_csv(filepath)


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_ORDER + [TARGET_COL])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f'Missing required dataset columns: {missing_columns}')


def preprocess_data(df):
    df = df.copy()
    validate_columns(df)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    return df, {'categorical_columns': CATEGORICAL_COLS, 'numerical_columns': NUMERICAL_COLS}


def prepare_features(df, target_col=TARGET_COL):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
