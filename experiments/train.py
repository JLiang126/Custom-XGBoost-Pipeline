import os
import mlflow
import csv
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from azure.storage.blob import BlobServiceClient

TRAIN_CSV_PATH = "train_data.csv"
TEST_CSV_PATH = "test_data.csv"

MLFLOW_TRACKING_URI = "http://mlflow-server.default.svc.cluster.local:1234"
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = "xgboost"

print("=== Loading XGBoost ===")
try: import my_xgboost # type: ignore
except ImportError: print("Error loading XGBoost cannot find the module")
print("    Loaded XGBoost\n")

print("=== Loading Dataset ===")
features, labels = load_breast_cancer(return_X_y=True)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=7)
print(f"    Dataset loaded. Training shape: {features_train.shape}, Test shape: {features_test.shape}\n")

# Saving loading data set to .csv as the booster I made can only read csv files
def save_to_csv(X_data, y_data, output_path):
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        for features, label in zip(X_data, y_data):
            row = list(features) + [label]
            writer.writerow(row)
            
    print(f"    Successfully created: {output_path}")

save_to_csv(features_train, labels_train, TRAIN_CSV_PATH)
save_to_csv(features_test, labels_test, TEST_CSV_PATH)

def main():

    print("=== Init MLFlow ===")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cpp-xgboost-on-k8s-cluster")

    with mlflow.start_run():
        params = {"num_trees": 100, "LR": 0.05, "max_depth": 2, "lambda": 0,"gamma": 0, "loss": my_xgboost.LogLoss()}
        mlflow_params = params.copy()
        mlflow_params["loss"] = "LogLoss"
        mlflow.log_params(mlflow_params);
        print("    Parameters Logged")
        
        print("=== Boosting ===")
        data = my_xgboost.DataMatrix(TRAIN_CSV_PATH)
        target = data.get_labels()
        model = my_xgboost.XGBoost(**params)
        model.train(data, target)
        
        test_data = pd.read_csv(TEST_CSV_PATH, header=None)
        features = test_data.iloc[:, :-1].values
        labels = test_data.iloc[:, -1].values
        
        predictions = [1 if model.predict(list(row)) > 0.0 else 0 for row in features]
        print(f"    Accuracy: {accuracy_score(labels, predictions)} | Recall: {recall_score(labels, predictions)} | F1: {f1_score(labels, predictions)}")

        os.makedirs("models", exist_ok=True)
        weights = "models/test.txt"
        model.save_model(weights)

        print("=== Saving to Azure ===")
        if AZURE_CONNECTION_STRING:
            try:
                blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
                container_client = blob_service_client.get_container_client(CONTAINER)

                if not container_client.exists():
                    container_client.create_container()

                blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=weights)

                with open(weights, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                print("Model saved to Azure Blob Storage")
            except Exception as e:
                print(f"Upload to Azure Blob Storage failed: {e}")
        else:
            print("Cannot access 'AZURE_STORAGE_CONNECTION_STRING' environment variable")

if __name__ == "__main__":
    main()
