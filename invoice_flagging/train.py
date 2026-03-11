from model_eval import train_random_forest, evaluate_classifier
from data_prepro import load_invoice_data, apply_labels, split_data, scale_features
import joblib
import os


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():

    # create models folder if it does not exist
    os.makedirs("../models", exist_ok=True)

    # load data
    df = load_invoice_data("../data/inventory.db")

    # apply risk labels
    df = apply_labels(df)

    # split dataset
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # scale features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "../models/scaler.pkl"
    )

    # train model
    grid_search = train_random_forest(X_train_scaled, y_train)

    # evaluate model
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest"
    )

    # save best model
    joblib.dump(
        grid_search.best_estimator_,
        "../models/random_forest_model.pkl"
    )

    print("\nModel saved to ../models/random_forest_model.pkl")


if __name__ == "__main__":
    main()