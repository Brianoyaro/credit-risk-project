from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve the project root independently of the working directory
BASE_DIR = Path(__file__).resolve().parent.parent
ECON_PATH = BASE_DIR / "data" / "economic_indicators.csv"

econ = pd.read_csv(ECON_PATH)
econ.columns = econ.columns.str.lower().str.replace(" ", "_", regex=False)

#####################################################################3
# Identify year columns
year_cols = [col for col in econ.columns if col.startswith('yr')]

# Convert wide to long format
econ_long = econ.melt(
    id_vars=['country', 'indicator'],
    value_vars=year_cols,
    var_name='year',
    value_name='economic_value'
)

# Extract numeric year
econ_long['year'] = econ_long['year'].str.replace('yr', '').astype(int)

# Pivot economic indicators into columns
econ_wide = econ_long.pivot_table(
    index=['country', 'year'],
    columns='indicator',
    values='economic_value',
    aggfunc='first'
).reset_index()

econ_wide.columns = econ_wide.columns.str.lower().str.replace(" ", "_", regex=False)

# Remove the columns index name
econ_wide.columns.name = None



#######################################################################################################3

class CreditFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, econ_wide_df=None):
        if econ_wide_df is None:
            self.econ_wide_df = econ_wide
        else:
            self.econ_wide_df = econ_wide_df

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_out = X.copy()

        # Clean and lower-case string variables if they exist
        for category in ["country", "loan_type", "new_versus_repeat"]:
            if category in X_out.columns:
                X_out[category] = X_out[category].astype(str).str.lower().str.replace(" ", "_", regex=False)

        # Date conversion and feature extraction
        for col in ['disbursement_date', 'due_date']:
            if col in X_out.columns:
                X_out[col] = pd.to_datetime(X_out[col], errors='coerce')

        if 'disbursement_date' in X_out.columns:
            X_out['loan_year'] = X_out['disbursement_date'].dt.year
            X_out['loan_month'] = X_out['disbursement_date'].dt.month
        else:
            # Default placeholders if date column is missing
            X_out['loan_year'] = 2022
            X_out['loan_month'] = 1

        # Financial ratios and logging
        if 'total_amount_to_repay' in X_out.columns and 'total_amount' in X_out.columns:
            X_out["interest_amount"] = X_out["total_amount_to_repay"] - X_out["total_amount"]
            X_out["repayment_ratio"] = X_out["total_amount_to_repay"] / X_out["total_amount"]
        else:
            X_out["interest_amount"] = 0
            X_out["repayment_ratio"] = 1.0

        if 'duration' in X_out.columns:
            X_out["duration_log"] = np.log1p(X_out["duration"])
        if 'total_amount' in X_out.columns:
            X_out["total_amount_log"] = np.log1p(X_out["total_amount"])

        # Merge economic indicators
        X_out = X_out.merge(
            self.econ_wide_df,
            left_on=['country', 'loan_year'],
            right_on=['country', 'year'],
            how='left'
        )
        if 'year' in X_out.columns:
            X_out.drop(columns=['year'], inplace=True)

        return X_out
