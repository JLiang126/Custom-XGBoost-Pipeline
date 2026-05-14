import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

def generate_synthetic_data():
    print("Generating non-linear synthetic data...")
    
    # 1. Generate 10,000 samples. 
    # n_features=5 gives us exactly the 5 interacting features.
    # noise=1.0 adds standard gaussian noise to make it realistic.
    X, y = make_friedman1(n_samples=10000, n_features=5, noise=1.0, random_state=42)

    # 2. Put it into a Pandas DataFrame
    feature_names = [f"Feature_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    # 3. Split into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Save to CSV files (index=False keeps it clean for C++)
    train_df.to_csv("synthetic_train.csv", index=False)
    test_df.to_csv("synthetic_test.csv", index=False)

    print(f"Successfully saved 'synthetic_train.csv' ({len(train_df)} rows)")
    print(f"Successfully saved 'synthetic_test.csv' ({len(test_df)} rows)")
    
    # Show a quick preview
    print("\nSample Data (First 3 rows):")
    print(train_df.head(3).round(4))

if __name__ == "__main__":
    generate_synthetic_data()