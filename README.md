# Time Series Comparison Models

This repository contains the implementation and evaluation of multiple multivariate time series (MTS) forecasting models applied to real-world datasets. The project aims to systematically compare baseline statistical models, deep learning architectures, and recent state-of-the-art approaches, highlighting their trade-offs in terms of accuracy, interpretability, and computational cost.

The study follows a consistent methodology across datasets:

1. Apply models in parallel using the same preprocessing and training setup.
2. Evaluate performance using standard metrics (MSE, RMSE, MAE, MAPE, R², etc.).
3. Compare results to identify which models are best suited for specific business or economic scenarios.

**Models included:**

* **Baselines**: Random Walk, ARIMA, SARIMAX
* **Deep Learning**: LSTM, TCN, LightGBM, N-BEATS
* **Recent Advances**: PatchTST, FEAT, GCN, GAT
* **Foundation Models** for Time Series: GPT4TS, Lag-Llama, AutoMixer, UniTime

This repository is intended as a resource for researchers and practitioners to explore the evolving landscape of time series forecasting and understand which models are most appropriate depending on the nature of the series and business requirements.
