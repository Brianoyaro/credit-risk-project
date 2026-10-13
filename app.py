import re
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "xgboost_pipeline.joblib"
ECON_PATH = BASE_DIR / "economic_indicators.csv"


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
numerical_cols = bundle["numerical_cols"]
categorical_cols = bundle["categorical_cols"]

feature_cols = numerical_cols + categorical_cols


# --------------------------------------------------
# Prepare economic indicators
# --------------------------------------------------

econ = pd.read_csv(ECON_PATH)

econ.columns = (
    econ.columns.str.lower()
    .str.replace(" ", "_", regex=False)
)

for col in ["country", "indicator"]:
    econ[col] = (
        econ[col]
        .astype("string")
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

year_cols = [
    col for col in econ.columns
    if col.startswith("yr")
]

if not year_cols:
    raise ValueError(
        "No economic year columns found, such as yr2022."
    )

econ_long = econ.melt(
    id_vars=["country", "indicator"],
    value_vars=year_cols,
    var_name="year",
    value_name="economic_value",
)

econ_long["year"] = (
    econ_long["year"]
    .str.replace("yr", "", regex=False)
    .astype(int)
)

econ_wide = econ_long.pivot_table(
    index=["country", "year"],
    columns="indicator",
    values="economic_value",
    aggfunc="first",
).reset_index()

econ_wide.columns.name = None


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Loan Prediction API",
    description=(
        "Predict loan outcomes from raw loan records "
        "using XGBoost and economic indicators."
    ),
    version="1.0.0",
)



class LoanRecord(BaseModel):
    # model_config = ConfigDict(extra="allow")

    ID: str | None = None
    customer_id: int | None = None
    country_id: str
    tbl_loan_id: int | None = None
    lender_id: int | None = None
    loan_type: str
    Total_Amount: float
    Total_Amount_to_Repay: float
    disbursement_date: str
    due_date: str | None = None
    duration: float
    New_versus_Repeat: str
    Amount_Funded_By_Lender: float
    Lender_portion_Funded: float
    Lender_portion_to_be_repaid: float

class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Predicted loan outcome (0 or 1)",
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Predicted probability for each target class",
    )

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_column_name(name: str) -> str:
    """Match the column normalization used during training."""
    return name.strip().lower().replace(" ", "_")


def normalize_category(value: Any) -> Any:
    """Normalize categories using the training convention."""
    if isinstance(value, str):
        return value.lower().replace(" ", "_")
    return value


def build_features(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Convert one raw CSV-like record into the exact
    engineered features expected by the saved pipeline.
    """

    # Normalize incoming CSV headers.
    normalized = {
        normalize_column_name(key): value
        for key, value in raw.items()
    }

    if "target" in normalized:
        raise HTTPException(
            status_code=422,
            detail="Do not include the target column in the request.",
        )

    # Required raw values for feature engineering.
    required = [
        "country_id",
        "loan_type",
        "new_versus_repeat",
        "disbursement_date",
        "duration",
        "total_amount",
        "total_amount_to_repay",
        "amount_funded_by_lender",
        "lender_portion_funded",
        "lender_portion_to_be_repaid",
    ]

    missing = [
        col for col in required
        if col not in normalized or normalized[col] is None
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required raw fields.",
                "missing_fields": missing,
            },
        )

    row = pd.DataFrame([normalized])

    # Convert raw dates.
    for col in ["disbursement_date", "due_date"]:
        if col in row.columns:
            row[col] = pd.to_datetime(
                row[col],
                errors="coerce",
            )

    if pd.isna(row.loc[0, "disbursement_date"]):
        raise HTTPException(
            status_code=422,
            detail="disbursement_date must be a valid date.",
        )

    # Normalize categorical values.
    for col in [
        "country_id",
        "loan_type",
        "new_versus_repeat",
    ]:
        row[col] = row[col].apply(normalize_category)

    # Rename country_id to country, as in training.
    row = row.rename(
        columns={"country_id": "country"}
    )

    # Date-derived features.
    row["loan_year"] = (
        row["disbursement_date"].dt.year
    )
    row["loan_month"] = (
        row["disbursement_date"].dt.month
    )

    # Numeric raw fields.
    numeric_raw_cols = [
        "duration",
        "total_amount",
        "total_amount_to_repay",
        "amount_funded_by_lender",
        "lender_portion_funded",
        "lender_portion_to_be_repaid",
    ]

    for col in numeric_raw_cols:
        row[col] = pd.to_numeric(
            row[col],
            errors="coerce",
        )

    # Financial features. Match training formulas.
    row["interest_amount"] = (
        row["total_amount_to_repay"]
        - row["total_amount"]
    )

    row["repayment_ratio"] = (
        row["total_amount_to_repay"]
        / row["total_amount"].replace(0, np.nan)
    )

    row["duration_log"] = np.log1p(
        row["duration"].where(row["duration"] >= 0)
    )

    row["total_amount_log"] = np.log1p(
        row["total_amount"].where(
            row["total_amount"] >= 0
        )
    )

    # Merge economic indicators using country and year.
    row = row.merge(
        econ_wide,
        left_on=["country", "loan_year"],
        right_on=["country", "year"],
        how="left",
        validate="many_to_one",
    )

    row = row.drop(
        columns=["year"],
        errors="ignore",
    )

    # Check that the saved feature definitions are valid.
    missing_model_features = [
        col for col in feature_cols
        if col not in row.columns
    ]

    if missing_model_features:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Feature engineering does not match "
                    "the saved model."
                ),
                "missing_model_features": missing_model_features,
            },
        )

    # Return exactly the training feature order.
    X = row[feature_cols].copy()

    # Numeric conversion and invalid value handling.
    for col in numerical_cols:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X[numerical_cols] = X[numerical_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Loan Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LoanRecord) -> PredictionResponse:
    """
    Accept a single loan record in JSON format and return
    the predicted outcome and probabilities.
    """

    # Convert Pydantic model to a dictionary
    raw = payload.model_dump()

    # Build the features expected by the trained pipeline
    X = build_features(raw)

    try:
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        return PredictionResponse(
            prediction=prediction,
            probabilities={
                str(label): float(probability)
                for label, probability
                in zip(model.classes_, probabilities)
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        import logging
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        ) from exc
