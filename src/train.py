import numpy as np
from utils import (
    load_dataset, create_model, save_model,
    calculate_metrics
)
import os
from sklearn.metrics import mean_absolute_error

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024

def main():
    print("Loading California Housing dataset.")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Creating LinearRegression model.")
    model = create_model()

    print("Training model.")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2, mse = calculate_metrics(y_test, y_pred)
    max_error = np.abs(y_test - y_pred).max()
    mean_error = np.abs(y_test - y_pred).mean()

    model._dummy_metadata = "A" * 1024 

    model_path = "models/linear_regression_model.joblib"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    model_size_kb = get_file_size_kb(model_path)

    print("\n Model Evaluation")
    print(f"R² Score:             {r2:.4f}")
    print(f"MSE:                  {mse:.4f}")
    print(f"Model Size:           {model_size_kb:.1f} KB")

    return model, r2, mse

if __name__ == "__main__":
    main()
