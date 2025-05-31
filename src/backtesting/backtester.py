import pandas as pd
import yaml
from pathlib import Path

# Import strategy execution modules
from src.strategy_execution.signal_combiner import combine_signals
from src.strategy_execution.portfolio_constructor import construct_portfolio

# Import execution simulation module (to be implemented)
# from src.execution_simulator import simulate_execution

class Backtester:
    def __init__(self, price_data, signals, strategy_config, backtest_config):
        self.price_data = price_data
        self.signals = signals
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config
        self.dates = price_data.index.unique()
        self.portfolio_value = []
        self.daily_returns = []
        self.trade_log = []
        self.cash = backtest_config['initial_cash']
        self.holdings = {}  # {ticker: shares}

    def run(self):
        for date in self.dates:
            # 1. Get today's signals and prices
            todays_signals = self.signals.loc[date]
            todays_prices = self.price_data.loc[date]

            # 2. Combine signals (ML + other signals)
            combined_signals = combine_signals(todays_signals, self.strategy_config)

            # 3. Construct portfolio (determine target positions)
            target_portfolio = construct_portfolio(combined_signals, self.strategy_config)

            # 4. Simulate order execution (apply transaction costs, slippage, liquidity)
            # executed_trades, self.cash, self.holdings = simulate_execution(
            #     self.holdings, target_portfolio, todays_prices, self.cash, self.backtest_config
            # )

            # 5. Update portfolio value
            portfolio_val = self.cash + sum(
                self.holdings.get(ticker, 0) * todays_prices.get(ticker, 0)
                for ticker in self.holdings
            )
            self.portfolio_value.append(portfolio_val)

            # 6. Calculate and store daily return
            if len(self.portfolio_value) > 1:
                daily_ret = (self.portfolio_value[-1] - self.portfolio_value[-2]) / self.portfolio_value[-2]
            else:
                daily_ret = 0.0
            self.daily_returns.append(daily_ret)

            # 7. Record trades (optional)
            # self.trade_log.extend(executed_trades)

    def results(self):
        return pd.DataFrame({
            'date': self.dates,
            'portfolio_value': self.portfolio_value,
            'daily_return': self.daily_returns
        }).set_index('date')

    # Optionally: add method to export trade log

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs
    strategy_config = load_yaml_config('config/strategy_config.yaml')
    backtest_config = load_yaml_config('config/backtest_config.yaml')

    # Load historical price data and signals (implement as needed)
    price_data = pd.read_csv('data/price_data.csv', index_col='date', parse_dates=True)
    signals = pd.read_csv('data/signals.csv', index_col='date', parse_dates=True)

    # Run backtest
    backtester = Backtester(price_data, signals, strategy_config, backtest_config)
    backtester.run()
    results = backtester.results()
    results.to_csv('output/backtest_results.csv')

if __name__ == "__main__":
    main()