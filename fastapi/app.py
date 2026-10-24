from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib
import uvicorn
import sys
from pathlib import Path


app = FastAPI(
    title="Credit Risk Assessment Engine",
    description="Serves XGBoost default predictions on raw credit inputs utilizing custom SKLearn transformers.",
    version="1.0.0"
)

# 1. Define the incoming raw loan request schema matching your upstream dataset structures
class LoanRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="ID")
    customer_id: int
    country: str
    tbl_loan_id: int
    lender_id: int
    loan_type: str
    total_amount: float = Field(..., alias="Total_Amount")
    total_amount_to_repay: float = Field(..., alias="Total_Amount_to_Repay")
    disbursement_date: str
    due_date: str
    duration: int
    new_versus_repeat: str = Field(..., alias="New_versus_Repeat")
    amount_funded_by_lender: float = Field(..., alias="Amount_Funded_By_Lender")
    lender_portion_funded: float = Field(..., alias="Lender_portion_Funded")
    lender_portion_to_be_repaid: float = Field(..., alias="Lender_portion_to_be_repaid")


# 2. Load the integrated pipeline containing the model on startup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_pipeline.joblib"
sys.path.insert(0, str(BASE_DIR / "notebooks"))
from CreditFeatureExtractor import CreditFeatureExtractor

try:
    artifacts = joblib.load(MODEL_PATH)
    model_pipeline = artifacts["model"]
    print("Integrated modeling pipeline successfully loaded.")
except Exception as e:
    raise RuntimeError(f"Failed to load modeling pipeline artifact: {str(e)}")

@app.post("/predict")
def predict_credit_risk(payload: LoanRequest):
    try:
        # Convert payload directly to a dictionary structure matching the raw schema
        input_dict = payload.model_dump(by_alias=True)

        # Convert key headers back to lowercase/snake_case to match the exact headers your Pipeline expects
        raw_df = pd.DataFrame([input_dict])
        raw_df.columns = raw_df.columns.str.lower().str.replace(" ", "_", regex=False)
        # raw_df.rename(columns={'country_id': 'country'}, inplace=True)

        # Feed the raw dataframe directly into the pipeline
        probability_of_default = float(model_pipeline.predict_proba(raw_df)[0][1])
        prediction = int(model_pipeline.predict(raw_df)[0])

        return {
            "loan_id": payload.id,
            "default_prediction": prediction,
            "probability_of_default": round(probability_of_default, 4),
            "status": "approved" if probability_of_default < 0.5 else "rejected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction generation failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)