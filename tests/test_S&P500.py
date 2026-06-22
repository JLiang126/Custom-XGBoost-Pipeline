import my_xgboost # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test() -> int:
    loss = my_xgboost.MSELoss()
    model = my_xgboost.XGBoost(0, 0.0, 0, 0.0, 0.0, 0.0, loss)
    
    model.load_model("models/sp500_model.txt")
    print("Model loaded")

    data = pd.read_csv("data/sp500_test.csv")
    features = data.iloc[:, :-1].values 
    returns = data.iloc[:, -1].values 
    
    def run_backtest(model, test_features, actual_returns):
        preds = np.array([model.predict(list(row)) for row in test_features])
        
        # Convert to signals (-1 or 1)
        signals = np.sign(preds)
        
        # Calculate Strategy Returns
        strategy_returns = signals * actual_returns
        
        # Calculate Cumulative Wealth starting at 1.0 and multiply by (1 + return) daily
        cumulative_strategy = np.cumprod(1 + strategy_returns)
        cumulative_market = np.cumprod(1 + actual_returns)
        
        total_return = (cumulative_strategy[-1] - 1) * 100
        win_rate = np.mean(strategy_returns > 0)
        
        # Standard Sharpe Ratio calculation
        std_dev = np.std(strategy_returns)
        if std_dev > 0:
            sharpe_ratio = (np.mean(strategy_returns) / std_dev) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        print(f"--- Backtest Results ---")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_strategy, label='XGBoost Strategy')
        plt.plot(cumulative_market, label='S&P 500 Buy & Hold', alpha=0.6)
        plt.title('Equity Curve: C++ XGBoost vs Market')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Multiple (1.0 = Start)')
        plt.legend()
        plt.grid(True)
        plt.show()

    run_backtest(model, features, returns)
    return 0

if __name__ == "__main__":
    test()