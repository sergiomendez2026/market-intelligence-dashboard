
# Methodology

## Project Objective

The Market Intelligence Dashboard analyzes financial assets using historical market data, technical indicators and machine learning models.

## Data Source

Market data is obtained through yFinance. The application currently supports stocks, cryptocurrencies, indices and forex pairs.

## Technical Indicators

The dashboard computes:

- Moving Average 20 days
- Moving Average 50 days
- Bollinger Bands
- RSI 14 days

## Machine Learning Approach

The model uses XGBoost to estimate the next-period price using historical price-based features and technical indicators.

## Validation Strategy

The dataset is split using temporal order. The first 80% of observations are used for training and the last 20% for testing.

The model is compared against a naïve baseline:

```text
prediction_next_period = current_price
