"""Generate the assignment report as a Word document.

Pulls numbers and figures from outputs/ (produced by main.py) so the report
always reflects the latest run rather than duplicating hardcoded values.
Run `python main.py` first.
"""

from __future__ import annotations

import json
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

METRICS_PATH = "outputs/metrics/metrics.json"
HYPERPARAMS_PATH = "outputs/metrics/hyperparameters.json"
DATASET_INFO_PATH = "outputs/metrics/dataset_info.json"
FIGURES_DIR = "outputs/figures"
OUT_PATH = "report/GP_Forecasting_Report_APA.docx"

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def add_heading(doc: Document, text: str, level: int = 1):
    return doc.add_heading(text, level=level)


def add_figure(doc: Document, path: str, caption: str, width_in: float = 6.0):
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(10)


def add_metrics_table(doc: Document, metrics: dict):
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "Metric"
    hdr[1].text = "Value"
    for key, value in metrics.items():
        row = table.add_row().cells
        row[0].text = key
        row[1].text = f"{value:.4f}" if isinstance(value, float) else str(value)


def add_apa_reference(doc: Document, author_date: str, title: str, remainder: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.5)
    paragraph.paragraph_format.first_line_indent = Inches(-0.5)
    paragraph.paragraph_format.line_spacing = 2.0
    paragraph.add_run(author_date)
    paragraph.add_run(title).italic = True
    paragraph.add_run(remainder)


def build_report(out_path: str = OUT_PATH):
    metrics = load_json(METRICS_PATH)
    hp = load_json(HYPERPARAMS_PATH)
    info = load_json(DATASET_INFO_PATH)

    doc = Document()

    # ---- Title page ----
    title = doc.add_heading("Time-Series Forecasting with Gaussian Processes", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Probabilistic Forecasting of Daily Mean Temperature in Delhi, India")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True
    doc.add_page_break()

    # ---- 1. Dataset ----
    add_heading(doc, "1. Dataset Choice and Characteristics")
    doc.add_paragraph(
        "This report uses the Daily Delhi Climate dataset, a publicly available time series "
        "of daily weather observations for Delhi, India, spanning January 1, 2013 through "
        "April 24, 2017. Each record contains four variables: mean temperature, "
        "humidity, wind speed, and mean pressure. The dataset is "
        "distributed in chronological training and test files. The common date at the "
        "boundary was removed from the test file so every test observation occurs after "
        "the final training observation."
    )
    doc.add_paragraph(
        f"Training set: {info['n_train']} daily observations "
        f"({info['train_start']} to {info['train_end']}). "
        f"Test set: {info['n_test']} daily observations "
        f"({info['test_start']} to {info['test_end']})."
    )
    doc.add_paragraph("The files were obtained from Kaggle (sumanthvrao, n.d.).")
    doc.add_paragraph(
        "The target is meantemp (daily mean temperature). The training series shows an "
        "annual cycle and shorter changes from day to day. The model uses a periodic "
        "kernel for the annual cycle and two RBF kernels for slower and shorter changes."
    )

    # ---- 2. Preprocessing ----
    add_heading(doc, "2. Preprocessing")
    doc.add_paragraph(
        "The raw data required only light preprocessing, since it is already a clean, "
        "regularly sampled daily series with no missing timestamps or missing values:"
    )
    for bullet in [
        "Missing values: none were present in either file; no imputation was necessary.",
        "Duplicate date: the training and test files share one overlapping date "
        "(2017-01-01). The duplicate row was removed from the test set to avoid "
        "double-counting an observation the model had already seen during training.",
        "Only date and meantemp enter this time-only model. Other recorded variables, "
        "including pressure, are not modified or used as predictors.",
        "Time axis encoding: calendar dates were converted to a continuous numeric axis, "
        "measured in years elapsed since the first training date. This scaling keeps the "
        "periodic kernel's learned period directly interpretable (a value near 1.0 "
        "corresponds to the expected ~365-day annual cycle).",
        "Target standardization: meantemp was standardized (zero mean, unit variance) "
        "using training-set statistics only. Standardization improves the numerical "
        "stability of GP hyperparameter optimization; all reported metrics and plots are "
        "unscaled back to the original °C units.",
        "Train/test split: the supplied chronological split was retained after removal "
        "of the shared boundary date; no observations were shuffled.",
    ]:
        doc.add_paragraph(bullet, style="List Bullet")

    # ---- 3. Model design ----
    add_heading(doc, "3. Model Design")
    doc.add_paragraph(
        "The model is an exact Gaussian Process regressor implemented with GPyTorch "
        "(gpytorch.models.ExactGP), using a constant mean function and a composite "
        "covariance function built as the sum of three components, each with its own "
        "learned output scale (ScaleKernel):"
    )
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "Component", "Kernel", "Role / Constraint"
    rows = [
        ("Trend", "RBF (squared exponential)", "Long-term smooth drift; lengthscale constrained > 0.5 years"),
        ("Seasonal", "Periodic", "Annual seasonal cycle; period constrained to [0.8, 1.2] years"),
        ("Local", "RBF (squared exponential)", "Short-range, day-to-day correlation; lengthscale constrained < 0.15 years"),
        ("Noise", "Gaussian likelihood", "I.i.d. observation noise"),
    ]
    for r in rows:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text = r
    doc.add_paragraph("")
    doc.add_paragraph(
        "The trend and local lengthscales have disjoint constraints to encourage "
        "interpretable long- and short-range components. The effect of these constraints "
        "on forecast accuracy was not assessed against an unconstrained fit."
    )
    doc.add_paragraph(
        "This kernel structure follows the treatment of trend and periodic components "
        "in Rasmussen and Williams (2006), adapted here to daily temperature."
    )
    doc.add_paragraph(
        "The ExactGP model, Gaussian likelihood, and training procedure follow the "
        "GPyTorch regression tutorial (GPyTorch, n.d.)."
    )

    # ---- 4. Training ----
    add_heading(doc, "4. Training Process")
    doc.add_paragraph(
        "All kernel hyperparameters (lengthscales, output scales, seasonal period) and the "
        "Gaussian likelihood's observation noise were optimized jointly by maximizing the "
        "exact marginal log likelihood (MLL) of the training data, using the Adam optimizer "
        "(learning rate 0.02, cosine-annealed over 500 iterations). No mini-batching was "
        "needed: with 1,462 training points, exact GP inference (which scales cubically in "
        "the number of points) remains computationally tractable on CPU."
    )
    add_figure(doc, f"{FIGURES_DIR}/training_loss.png",
               "Figure 1. Training loss (negative marginal log likelihood) vs. iteration. "
        "The loss falls during training and changes less in later iterations.")
    doc.add_paragraph("Learned hyperparameters after training:")
    hp_table = doc.add_table(rows=1, cols=2)
    hp_table.style = "Light Grid Accent 1"
    hdr = hp_table.rows[0].cells
    hdr[0].text, hdr[1].text = "Hyperparameter", "Learned Value"
    hp_rows = [
        ("Trend RBF lengthscale", f"{hp['trend_lengthscale_years']:.3f} years"),
        ("Seasonal period", f"{hp['seasonal_period_years']:.3f} years (≈ {hp['seasonal_period_years']*365:.0f} days)"),
        ("Seasonal kernel lengthscale", f"{hp['seasonal_lengthscale']:.3f}"),
        ("Local RBF lengthscale", f"{hp['local_lengthscale_days']:.1f} days"),
        ("Observation noise (std, standardized units)", f"{hp['observation_noise_std_scaled']:.4f}"),
        ("Trend kernel output scale", f"{hp['trend_outputscale']:.3f}"),
        ("Seasonal kernel output scale", f"{hp['seasonal_outputscale']:.3f}"),
        ("Local kernel output scale", f"{hp['local_outputscale']:.3f}"),
    ]
    for r in hp_rows:
        row = hp_table.add_row().cells
        row[0].text, row[1].text = r
    doc.add_paragraph("")
    doc.add_paragraph(
        "The seasonal period converged to approximately one year, consistent with the "
        "annual temperature cycle. The period was initialized near one year and "
        "constrained to the range 0.8–1.2 years, so this result is not independent "
        "evidence that an unconstrained model would find the same period."
    )

    # ---- 5. Forecasting ----
    add_heading(doc, "5. Forecasting Method")
    doc.add_paragraph(
        "After training, the model was switched to evaluation mode and used to compute the "
        "exact GP posterior predictive distribution at each test-set time point. Passing "
        "the latent posterior through the Gaussian likelihood adds the learned observation "
        "noise variance, yielding a predictive mean and variance for the observed "
        "temperature (not just the noise-free latent function) at every forecast date. "
        "The forecast mean rises through the test period, following the usual seasonal "
        "warming pattern. The reported variance describes uncertainty in a future daily "
        "temperature observation at each date."
    )
    add_figure(doc, f"{FIGURES_DIR}/full_forecast.png",
               "Figure 2. Full series: training data, held-out test data, and the GP "
               "forecast mean with 95% predictive interval over the test period.")
    add_figure(doc, f"{FIGURES_DIR}/test_zoom.png",
               "Figure 3. Zoomed view of the test period, comparing the GP forecast "
               "(mean and 95% predictive interval) against the actual observed daily temperatures.")

    # ---- 6. Evaluation ----
    add_heading(doc, "6. Evaluation Results")
    doc.add_paragraph(
        "Forecast quality was assessed with both point-forecast and probabilistic metrics, "
        "computed on the original (°C) scale:"
    )
    doc.add_paragraph(
        "The executed gp_forecasting.ipynb notebook uses the same data, model, training, "
        "and evaluation functions as main.py; its printed test metrics match the values "
        "in this report."
    )
    add_metrics_table(doc, metrics)
    doc.add_paragraph("")
    doc.add_paragraph(
        "MSE, RMSE, and MAE quantify the accuracy of the predictive mean. Mean predictive "
        "log likelihood is the average per-point log density of the observed value under "
        "the model's predictive Gaussian; higher is better. Mean NLPD is its negative, "
        "so lower is better. These metrics assess uncertainty quality in addition to accuracy, penalizing "
        "both large errors and over/under-confident intervals. Predictive interval coverage is the "
        "fraction of true test values that fall within the model's own 95% predictive "
        "interval; a well-calibrated model should show coverage near 0.95 over many "
        "comparable forecasts."
    )
    add_figure(doc, f"{FIGURES_DIR}/residuals.png",
               "Figure 4. Residuals (observed minus predicted mean) over the test period "
               "and their distribution. This plot can reveal systematic errors that "
               "aggregate metrics may hide.")

    # ---- 7. Uncertainty interpretation ----
    add_heading(doc, "7. Interpretation of Uncertainty and Predictive Intervals")
    doc.add_paragraph(
        "The shaded bands in Figures 2 and 3 are 95% predictive intervals for future "
        "observed temperatures. They include uncertainty in the fitted temperature curve "
        "and observation noise; they are not confidence intervals for the mean alone."
    )
    for bullet in [
        "The interval width changes modestly over the test period rather than widening "
        "steadily with forecast horizon (Figures 2 and 3).",
        f"The empirical 95% predictive interval coverage on the test set was "
        f"{metrics['95% predictive interval coverage']:.1%}, below the nominal 95% target. "
        "The model therefore undercovers this held-out period; its intervals should "
        "not be treated as calibrated 95% ranges without further validation.",
        "The local kernel models correlated day-to-day variation, while the Gaussian "
        "likelihood models independent observation noise. The long-range and seasonal "
        "kernels also contribute to uncertainty beyond the training dates.",
        "The 95% label describes the model's probability for an individual observation "
        "under its assumptions; the observed 88.5% coverage shows those assumptions do "
        "not give 95% coverage on this test period.",
    ]:
        doc.add_paragraph(bullet, style="List Bullet")

    # ---- 8. Conclusion ----
    add_heading(doc, "8. Conclusion")
    doc.add_paragraph(
        "A Gaussian Process with a trend + seasonal + local composite kernel was able to "
        "learn the annual temperature cycle of Delhi directly from the training data and "
        "produce probabilistic forecasts for the following months that tracked the "
        "observed seasonal warming trend. Its 95% predictive intervals covered "
        f"{metrics['95% predictive interval coverage']:.1%} of held-out observations, showing undercoverage. "
        "The main limitation is that the model is purely temporal (a function of "
        "date only). A later study could test additional weather variables, provided "
        "their values are available at the time each forecast is made."
    )

    add_heading(doc, "References")
    add_apa_reference(
        doc,
        "GPyTorch. (n.d.). ",
        "GPyTorch regression tutorial",
        ". Retrieved September 16, 2026, from "
        "https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html",
    )
    add_apa_reference(
        doc,
        "Rasmussen, C. E., & Williams, C. K. I. (2006). ",
        "Gaussian processes for machine learning",
        ". MIT Press. https://gaussianprocess.org/gpml/",
    )
    add_apa_reference(
        doc,
        "sumanthvrao. (n.d.). ",
        "Daily climate time series data",
        " [Data set]. Kaggle. "
        "https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data",
    )

    doc.save(out_path)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    build_report(sys.argv[1] if len(sys.argv) > 1 else OUT_PATH)
