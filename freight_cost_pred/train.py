import joblib
from pathlib import Path

from data_prepro import load_vendor_invoice_data, prepare_features, split_data
from model_eval import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    db_path = "data/inventory.db/vendor_invoice.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    df = load_vendor_invoice_data(db_path)
    
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))

    best_model_info = min(results, key=lambda x: x["MAE"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]

    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\n Best model: {best_model_name} saved to {model_path}")
    print(f"\nModel path: {model_path}")

if __name__ == "__main__": main()