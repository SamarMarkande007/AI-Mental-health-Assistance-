# 🧠 AI Mental Health Assistance

> A collaborative NHS mental health data science pipeline - covering statistical analysis, machine learning, demand forecasting, anomaly detection, and geospatial visualisation across a structured 4-week sprint.

---

## 📌 Overview

This project analyses NHS mental health service data to surface actionable insights using classical statistics, supervised and unsupervised machine learning, time-series forecasting, and geographic analysis. It is a multi-contributor team project, with each intern owning a set of weekly deliverables integrated into a single reproducible pipeline.

---

## 👥 Contributors

| Folder | Contributor |
|---|---|
| `DS-1_Samar_Markande/` | Samar Markande |
| `DS-2_Anurag_Chawla/` | Anurag Chawla |
| `DS-3_Maya/` | Maya |
| `DS_4_Ajay/` | Ajay |
| `DS_4_Rushay/` | Rushay Gopani |

---

## 📁 Project Structure

```
AI-Mental-health-Assistance-/
│
├── DS-1_Samar_Markande/        # DS-1 notebooks & analysis
├── DS-2_Anurag_Chawla/         # DS-2 notebooks & analysis
├── DS-3_Maya/                  # DS-3 notebooks & analysis
├── DS_4_Ajay/                  # DS-4 notebooks & analysis
├── DS_4_Rushay/                # DS-4 notebooks & analysis
│
├── datasets/                   # Raw and processed input data
├── outputs/                    # All generated plots, CSVs, and logs
├── scripts/
│   ├── week1/                  # Week 1 analysis scripts
│   ├── week2/                  # Week 2 modelling scripts
│   └── week3/                  # Week 3 forecasting & spatial scripts
├── Reports/                    # Written reports and documentation
│
├── run_all.py                  # Master pipeline runner
├── ds2_mhsds_integration_into_pipeline.py   # Monthly data integration pipeline
├── requirements.txt
└── .gitignore
```

---

## 🔬 Pipeline Overview

The pipeline is structured as a 4-week sprint of progressively advanced analyses:

### Week 1 — Statistical Foundations
- **DS-1** · Statistical baseline modelling
- **DS-2** · Hypothesis testing
- **DS-3** · Distribution analysis

### Week 2 — Machine Learning
- **DS-1** · Regression modelling
- **DS-2** · Clustering & patient segmentation
- **DS-3** · Temporal pattern detection

### Week 3 — Forecasting & Intelligence
- **DS-1** · IAPT demand forecasting (Prophet)
- **DS-2** · Prescribing anomaly detection
- **DS-3** · Geospatial analysis

### Week 4 — Reproducibility & Validation
- **DS-1** · Full pipeline reproducibility testing & run logging
- **DS-2** · MHSDS data integration into pipeline
- **DS-3** · Results validation & cross-checks
- **DS-4** · Final reporting & output consolidation

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+

1. Clone the repository:
   ```bash
   git clone https://github.com/SamarMarkande007/AI-Mental-health-Assistance-.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Pipeline

To execute all scripts end-to-end from the project root:

```bash
python run_all.py
```

This will:
- Run all 9 analysis scripts across Weeks 1–3 in sequence
- Save all outputs (charts, CSVs, model results) to `outputs/`
- Generate a timestamped log file at `outputs/w4_ds1_reproducibility_test/pipeline_run_<timestamp>.log`
- Print a summary showing which scripts passed, failed, or were skipped

### Running individual scripts

```bash
python scripts/week1/ds1_statistical_baseline_model.py
python scripts/week2/ds2_clustering_patient_segments.py
python scripts/week3/ds3_geo_spatial_analysis.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `numpy` / `pandas` | Data manipulation |
| `scipy` / `statsmodels` | Statistical analysis |
| `scikit-learn` | Machine learning |
| `prophet` | Time-series forecasting |
| `matplotlib` / `seaborn` | Visualisation |
| `geopandas` | Geospatial analysis |
| `openpyxl` / `xlrd` | Excel file handling |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Outputs

All generated outputs are saved to the `outputs/` directory, including:
- Statistical test results and summary tables
- Regression model performance metrics
- Patient clustering visualisations
- IAPT demand forecast charts
- Prescribing anomaly reports
- Geospatial maps
- Full pipeline run logs

---

## 📝 Reports

Detailed written reports for each analysis phase are available in the `Reports/` directory.

---

## ⚠️ Data Note

This project uses NHS mental health service datasets. Ensure any data placed in `datasets/` complies with applicable data governance and NHS data sharing agreements before use.

---

## 📄 License

MIT License