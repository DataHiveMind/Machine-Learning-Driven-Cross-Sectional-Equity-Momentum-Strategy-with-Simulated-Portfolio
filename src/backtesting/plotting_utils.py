import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def plot_equity_curve(equity_curve: pd.Series, benchmark_curve: pd.Series = None, save_path: str = None):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Strategy')
    if benchmark_curve is not None:
        plt.plot(benchmark_curve, label='Benchmark', linestyle='--')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_daily_returns_histogram(daily_returns: pd.Series, save_path: str = None):
    plt.figure(figsize=(8, 5))
    sns.histplot(daily_returns, bins=50, kde=True)
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_drawdown(equity_curve: pd.Series, save_path: str = None):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_rolling_metric(daily_returns: pd.Series, window: int = 63, metric: str = 'sharpe', save_path: str = None):
    if metric == 'sharpe':
        rolling = daily_returns.rolling(window)
        rolling_sharpe = (rolling.mean() / rolling.std()) * np.sqrt(252)
        plt.figure(figsize=(12, 4))
        plt.plot(rolling_sharpe, label=f'{window}-Day Rolling Sharpe')
        plt.title(f'Rolling Sharpe Ratio ({window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
    elif metric == 'volatility':
        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
        plt.figure(figsize=(12, 4))
        plt.plot(rolling_vol, label=f'{window}-Day Rolling Volatility')
        plt.title(f'Rolling Volatility ({window} days)')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_heatmap(data: pd.DataFrame, save_path: str = None):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example for custom plot (e.g., feature importance)
def plot_feature_importance(importances: pd.Series, save_path: str = None):
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()