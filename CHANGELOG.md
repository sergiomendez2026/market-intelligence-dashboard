# Changelog

All notable changes to this project will be documented in this file.

---

## [v0.1.0] - 2026-06-17

### Added

- Interactive Market Intelligence Dashboard deployed on Streamlit Cloud.
- Modular architecture under `src/`.
- Financial data loading using public market data sources.
- Technical indicators:
  - Moving averages.
  - RSI.
  - Bollinger Bands.
  - Volatility indicators.
- Machine learning pipeline for financial time series.
- Directional classification model.
- Model comparison module:
  - Baseline Dummy.
  - Logistic Regression.
  - Random Forest.
  - XGBoost.
- Integrated Market Signal Score.
- Backtesting engine with:
  - Buy & Hold benchmark.
  - Strategy return.
  - Excess return.
  - Drawdown.
  - Exposure.
  - Commission.
  - Slippage.
  - Transaction cost estimation.
- Walk-forward validation for temporal robustness.
- Lightweight financial sentiment fallback.
- GitHub Actions workflow for automated Python testing.
- Unit tests for:
  - Financial metrics.
  - Technical indicators.
  - Market signals.
- Technical README documentation.
- GitHub Pages project website.
- Streamlit Cloud dashboard deployment.

### Methodological Notes

- The system does not provide financial advice.
- The models are evaluated against simple baselines.
- The project avoids overpromising predictive performance.
- Results depend on asset, period, market regime and data quality.
- Current sentiment module uses a lightweight fallback; FinBERT is planned as an advanced extension.

### Known Limitations

- No real-time broker execution.
- No order book or liquidity constraints.
- No tax optimization.
- No dynamic spread modeling.
- No full portfolio optimization yet.
- No complete FinBERT integration yet.
- No user authentication or SaaS layer yet.

### Next Milestones

- Add FinBERT-based financial sentiment.
- Add portfolio-level backtesting.
- Add hyperparameter optimization.
- Add model explainability.
- Add PDF report generation.
- Add multilingual interface.
- Add FastAPI backend.
- Add SaaS-ready authentication layer.
