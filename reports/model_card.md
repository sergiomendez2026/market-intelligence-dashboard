# Model Card - Market Intelligence Dashboard

## Model

XGBoost Regressor

## Task

Next-period price estimation for selected financial assets.

## Inputs

The model uses:

- Current price
- Moving averages
- RSI
- Rolling volatility
- Daily returns
- 5-day returns

## Output

Predicted next-period price.

## Evaluation

The model is evaluated using time-based train/test split and compared against a naïve baseline.

## Limitations

Financial markets are noisy, non-stationary and affected by external events. Historical performance does not guarantee future predictive performance.

## Intended Use

Educational, portfolio and analytical demonstration.

## Not Intended For

Automated trading, investment advice or financial decision-making without professional validation.
