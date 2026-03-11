import joblib
import pandas as pd

MODEL_PATH = "../models/random_forest_model.pkl"


def load_model(model_path=MODEL_PATH):
    """Load trained invoice flagging model."""
    model = joblib.load(model_path)
    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice fraud flag.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    predictions = model.predict(input_df)

    input_df["Predicted_Flag"] = predictions

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100, 80, 40, 120],
        "invoice_dollars": [18500, 9000, 3000, 15000],
        "Freight": [1200, 800, 300, 900],
        "total_item_quantity": [500, 300, 120, 600],
        "total_item_dollars": [25000, 14000, 5000, 30000]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)