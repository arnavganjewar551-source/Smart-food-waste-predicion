
import joblib
import pandas as pd
from pathlib import Path


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# 2. Load the trained model
model = joblib.load(MODEL_PATH)


# 3. Create a prediction function
def predict_demand(
    week,
    center_id,
    meal_id,
    checkout_price,
    base_price,
    emailer_for_promotion,
    homepage_featured,
    category,
    cuisine,
    city_code,
    region_code,
    center_type,
    op_area
):

    # Prepare input data
    input_data = pd.DataFrame([{
        "week": week,
        "center_id": center_id,
        "meal_id": meal_id,
        "checkout_price": checkout_price,
        "base_price": base_price,
        "emailer_for_promotion": emailer_for_promotion,
        "homepage_featured": homepage_featured,
        "category": category,
        "cuisine": cuisine,
        "city_code": city_code,
        "region_code": region_code,
        "center_type": center_type,
        "op_area": op_area
    }])

    # Predict demand
    prediction = model.predict(input_data)[0]

    # Orders cannot be negative
    prediction = max(0, prediction)

    return round(prediction)


# 4. Example prediction
if __name__ == "__main__":

    predicted_orders = predict_demand(
        week=150,
        center_id=10,
        meal_id=1885,
        checkout_price=150.0,
        base_price=160.0,
        emailer_for_promotion=1,
        homepage_featured=0,
        category="Beverages",
        cuisine="Thai",
        city_code=590,
        region_code=56,
        center_type="TYPE_A",
        op_area=2.5
    )

    print("\nFOOD DEMAND PREDICTION")
    print("---------------------")
    print("Predicted orders:", predicted_orders)
