import joblib
import pandas as pd

MODEL_PATH = "../models/random_forest_model.pkl"


def load_model(model_path=MODEL_PATH):
    """Load trained classifier model"""

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices

    Parameters:
    input_data: dict

    Returns:
    pd.DataFrame with predicted flag
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df).round()

    return input_df


if __name__ == "__main__":

    # Example new invoice data
    sample_input = {
        "invoice_quantity": [120],
        "invoice_dollars": [5000],
        "Freight": [120],
        "total_item_quantity": [118],
        "total_item_dollars": [4980]
    }

    result = predict_invoice_flag(sample_input)

    print("Prediction Result:")
    print(result)