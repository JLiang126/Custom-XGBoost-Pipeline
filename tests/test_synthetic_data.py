import my_xgboost # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_synthetic_model():
    
    loss = my_xgboost.MSELoss()
    model = my_xgboost.XGBoost(0, 0.0, 0, 0.0, 0.0, 0.0, loss)
    model.load_model("models/synthetic_model.txt")
    print("Model Loaded")

    test_data = pd.read_csv("data/synthetic_test.csv")
    features = test_data.iloc[:, :-1].values 
    labels = test_data.iloc[:, -1].values 
    print("Data Loaded")

    preds = np.array([model.predict(list(row)) for row in features])

    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))

    print(f"\n--- Synthetic Data Evaluation ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"RMSE:           {rmse:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.5, color='blue', label='Predictions')
    
    min_val = min(np.min(labels), np.min(preds))
    max_val = max(np.max(labels), np.max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.xlabel('Actual Target Values (y)')
    plt.ylabel('Model Predictions')
    plt.title(f'XGBoost Engine Validation\nR² Score: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_synthetic_model()