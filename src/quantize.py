import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error
from utils import (
    load_model,
    quantize_to_uint8,
    dequantize_from_uint8,
    quantize_to_uint8_individual,
    dequantize_from_uint8_individual,
    load_dataset
)

def main():
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")

    coef = model.coef_
    intercept = model.intercept_
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept:.6f}")
    print(f"Original coef values: {coef}")

    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_params, "models/unquant_params.joblib")

    quant_coef, coef_metadata = quantize_to_uint8_individual(coef)
    print(f"\nQuantizing intercept...")
    quant_intercept, int_min, int_max, int_scale = quantize_to_uint8(np.array([intercept]))
    print(f"Intercept scale factor: {int_scale:.2f}")

    quant_params = {
        'quant_coef': quant_coef,
        'coef_metadata': coef_metadata,
        'quant_intercept': quant_intercept[0],
        'int_min': int_min,
        'int_max': int_max,
        'int_scale': int_scale
    }
    quant_path = "models/quant_params.joblib"
    joblib.dump(quant_params, quant_path)
    print(f"Quantized parameters saved to {quant_path}")

    dequant_coef = dequantize_from_uint8_individual(quant_coef, coef_metadata)
    dequant_intercept = dequantize_from_uint8(np.array([quant_intercept[0]]), int_min, int_max, int_scale)[0]

    coef_error = np.abs(coef - dequant_coef).max()
    intercept_error = np.abs(intercept - dequant_intercept)
    print(f"Max coefficient error: {coef_error:.8f}")
    print(f"Intercept error: {intercept_error:.8f}")

    X_train, X_test, y_train, y_test = load_dataset()

    y_pred_original = model.predict(X_test)
    y_pred_manual = X_test @ coef + intercept
    y_pred_dequant = X_test @ dequant_coef + dequant_intercept

    r2 = r2_score(y_test, y_pred_dequant)
    mse = mean_squared_error(y_test, y_pred_dequant)
    max_pred_error = np.max(np.abs(y_test - y_pred_dequant))
    mean_pred_error = np.mean(np.abs(y_test - y_pred_dequant))

    model_size_kb = os.path.getsize(quant_path) / 1024

    print("\nEvaluation Metrics")
    print(f"R² Score:             {r2:.4f}")
    print(f"MSE:                  {mse:.4f}")
    print(f"Quantized Model Size: {model_size_kb:.1f} KB")

    print("\nInference Test (first 10 samples):")
    print(f"\nOriginal predictions (sklearn): {y_pred_original[:10]}")
    print(f"Manual original predictions:    {y_pred_manual[:10]}")
    print(f"Manual dequant predictions:     {y_pred_dequant[:10]}")
    print(f"\nSklearn vs manual original:     {np.abs(y_pred_original[:10] - y_pred_manual[:10])}")
    print(f"Original vs dequant manual:     {np.abs(y_pred_manual[:10] - y_pred_dequant[:10])}")

if __name__ == "__main__":
    main()
