# Multivariate Load Forecasting with Exogenous Weather Variables for Transformer Stations

**Topic:** Electricity Demand Forecasting

**Credit Load:** 6 ECTS (150 hours)


## Background and Motivation
Electricity consumption at transformer stations is not an isolated signal — it is systematically influenced by external factors, most notably weather. Temperature drives heating and cooling loads, solar irradiance determines the magnitude of photovoltaic (PV) feed-in that offsets grid demand, and cloud cover modulates both. For a municipal grid operator managing over 700 transformer stations, the ability to accurately forecast load by leveraging weather data alongside historical consumption records is a key step toward smarter grid operation.

Your work investigates how modern deep learning models for **multivariate time series forecasting** — particularly those designed to handle **exogenous (external) variables** — perform when applied to transformer station load data with co-located weather inputs. You will systematically compare multiple model architectures available in TSLib, assess which weather features contribute most to predictive accuracy, and benchmark conventional trained models against zero-shot foundation models that require no task-specific training data.

## The Time Series Library (TSLib)
TSLib is an open-source deep learning library developed at Tsinghua University, providing a unified benchmark and development environment for time series analysis tasks. The tasks supported by the library include long-term forecasting, short-term forecasting, imputation, anomaly detection, and classification — all accessible through a single entry point (`run.py`).
For your work, the most relevant components are:
* The **long-term and short-term forecasting** experiment pipelines (`exp/exp_long_term_forecasting.py`, `exp/exp_short_term_forecasting.py`)
* The **TimeXer** model, which defines TSLib's practical paradigm for forecasting with exogenous variables (NeurIPS 2024)
* The **zero-shot forecasting** pipeline (`exp/exp_zero_shot_forecasting.py`), which enables evaluation of large pre-trained time series foundation models without any training on your dataset
* The `data_provider/` module, which handles data loading, windowing, and train/validation/test splits
You are expected to develop a solid working knowledge of the library's architecture, configuration system, and evaluation workflow.

## Dataset
You will be provided with a **synthetic dataset** simulating transformer station load measurements together with co-located **weather variables** (e.g., temperature, solar irradiance, cloud cover). The dataset is designed to reflect realistic trafo load characteristics from a municipal grid, including seasonal patterns, daily demand cycles, and PV-driven load reductions during daylight hours. Full variable descriptions and data format specifications will be provided separately.

## Your Task
Your work is organized into three connected phases:

**Phase 1 — Multivariate Forecasting Model Benchmarking**

You will implement and evaluate at least four forecasting models from TSLib across two prediction horizons: short-term (24–48 hours ahead) and long-term (up to 7 days ahead). The required models are:
* **TimeXer** — Transformer-based model with native support for exogenous variables (your primary model of interest)
* **iTransformer** — Inverted-attention Transformer, strong state-of-the-art baseline for multivariate forecasting
* **TimeMixer** — Decomposable multiscale MLP-based model
* **DLinear** — Simple linear model serving as a sanity-check baseline
For each model and horizon combination, report MSE and MAE on the test set. Present results in a structured comparison table.

**Phase 2 — Weather Feature Ablation Study**

To understand which weather variables actually contribute to forecast accuracy, you will conduct a systematic **ablation study**: starting from the full multivariate input (load + all weather features), you will retrain the best-performing model from Phase 1 while removing one weather variable at a time. The change in forecasting error (ΔMSE, ΔMAE) for each removed feature will indicate its marginal predictive value. This analysis will provide practical guidance on which sensor data is worth collecting and maintaining for real-world deployment.

**Phase 3 — Zero-Shot Forecasting Evaluation**

TSLib supports evaluation of large pre-trained time series foundation models (Large Time Series Models, LTSMs) in a **zero-shot** setting — meaning the model is applied directly to your dataset without any training. You will evaluate at least **one LTSM** (suggested: Chronos or TiRex) on the same forecasting tasks from Phase 1 and compare its performance against the trained models. This addresses a practically important question: for a grid operator onboarding a new trafo station with no historical data, how useful is a zero-shot foundation model as a cold-start solution?

## Deliverables
You are expected to submit the following:
1. **Written Report** — A structured scientific report (~4,000–6,000 words) covering:
    * Introduction and motivation
    * Related work (brief literature review on multivariate load forecasting and weather-driven demand models)
    * Description of the dataset and feature set
    * Methodology: models used, forecasting horizons, experimental setup, evaluation metrics
    * Results: Phase 1 benchmarking tables, Phase 2 ablation results, Phase 3 zero-shot comparison
    * Discussion: interpretation of results, limitations, and implications for the grid operator use case
    * Conclusion
2. **Code Repository** — A clean, documented version of all experimental scripts, structured for reproducibility and based on the TSLib framework.
3. **Presentation** — A 15–20 minute oral presentation of your findings, including discussion of model behavior and the practical relevance of your results for real-world trafo load forecasting.

## Recommended Starting Points
* Work through the TSLib tutorial notebook (`tutorial/TimesNet_tutorial.ipynb`) to get familiar with the framework before running your own experiments
* Study the `exp/exp_long_term_forecasting.py` file and the example scripts under `scripts/long_term_forecast/` to understand how forecasting experiments are configured
* Read the TimeXer paper (NeurIPS 2024) to understand the exogenous variable forecasting paradigm before designing your experiments
* Review the zero-shot forecasting scripts and the documentation for Chronos or TiRex before Phase 3
* Familiarize yourself with the evaluation metrics in `utils/metrics.py` (MSE, MAE, and optionally DTW)
