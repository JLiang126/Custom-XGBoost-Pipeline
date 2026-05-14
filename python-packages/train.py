import my_xgboost # type: ignore
import pandas as pd
import numpy as np
import time

def train() -> int:

    data = my_xgboost.DataMatrix("data/sp500_train.csv")
    print("Data Loaded")
    print(f"Engine Data Check:")

    row = data.get_row(0)
    column = data.get_column(0)
    target = data.get_labels()

    print(f" - Rows in C++: {data.get_num_rows()} First Row: {row}")
    print(f" - Features in C++: {data.get_num_columns()} First Column {column[:3]}")
    print(f" - First 3 Labels: {target[:3]}")

    loss = my_xgboost.MSELoss()

    # num_trees, learning_rate, max_depth, lambda, gamma, min_cover, objective

    # # Friedman1 Hyperparams
    # model = my_xgboost.XGBoost(500, 0.05, 4, 1.0, 0.0, 1.0, loss)
    # S&P500 Hyperparams
    model = my_xgboost.XGBoost(1000, 0.01, 4, 10.0, 0.1, 15.0, loss) 
    print("Boost Initialised")

    go = input("Train? ")
    if go not in ["Y", "Yes", "y", "YES"]: 
        print("Abort Training")
        return 0

    print("Training")
    start_time = time.time()
    model.train(data, target)
    end_time = time.time()

    print(f"Training Complete in {end_time - start_time:.4f} seconds!")

    model.save_model("models/sp500_model.txt")
    print("Model saved")
    return 0

train()