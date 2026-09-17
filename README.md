# Time-Series Forecasting with Gaussian Processes

Probabilistic forecasting of Delhi's daily mean temperature using an exact
Gaussian Process regression model (GPyTorch), with a composite kernel
(trend + annual seasonality + short-range variation) and
predictive uncertainty.

Dataset: [Daily Climate time series data on Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data).

## Project structure

```
DailyDelhiClimateTrain.csv   Training data (2013-01-01 to 2017-01-01)
DailyDelhiClimateTest.csv    Test data (2017-01-01 to 2017-04-24)
src/
  data.py                    Loading, cleaning, and standardization
  model.py                   GP model + composite kernel definition
  train.py                   Training loop (MLL optimization)
  evaluate.py                MSE/RMSE/MAE/NLPD/coverage metrics
  plotting.py                Forecast, loss, and residual plots
main.py                      End-to-end script: run the full pipeline
gp_forecasting.ipynb         Same pipeline as an annotated notebook
generate_report.py           Builds report/GP_Forecasting_Report_APA.docx
outputs/                     Generated metrics (JSON) and figures (PNG)
report/                      Generated .docx report for submission
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline (trains the model, forecasts, evaluates, saves plots
and metrics to `outputs/`):

```bash
python main.py
```

Generate the written report (reads `outputs/`, must be run after `main.py`):

```bash
python generate_report.py
```

Or explore interactively:

```bash
jupyter notebook gp_forecasting.ipynb
```

## Method summary

- **Target:** daily mean temperature (`meantemp`), standardized using
  training-set statistics.
- **Kernel:** `ScaleKernel(RBF)` for long-term trend + `ScaleKernel(Periodic)`
  for the annual cycle + `ScaleKernel(RBF)` for short-range/day-to-day
  correlation, plus a `GaussianLikelihood` noise term.
- **Training:** exact marginal log-likelihood maximization via Adam
  (500 iterations, cosine-annealed learning rate).
- **Evaluation:** MSE, RMSE, MAE for point-forecast accuracy; mean negative
  log predictive density and 95% predictive interval coverage for uncertainty assessment.

The held-out 95% interval coverage is below the nominal 95% target. See the
report for the measured result and limitations. No separate assignment rubric
was provided with this project, so the report follows the criteria in the
assignment instructions.

See `report/GP_Forecasting_Report_APA.docx` for the full write-up.
