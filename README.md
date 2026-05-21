# 👥 HR Attrition & Employee Risk Analyzer

> Predict which employees are likely to leave — and explain exactly why — using machine learning and SHAP explainability. Built end-to-end with CatBoost, Optuna, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange) ![Streamlit](https://img.shields.io/badge/App-Streamlit-red) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

---

## 🎯 Project Overview

Employee attrition costs companies **6–9 months of salary** per lost employee. This project builds an end-to-end system that:

- Predicts each employee's probability of leaving
- Explains the top reasons driving that risk
- Provides a live dashboard for HR teams to monitor the entire workforce

Built on the **IBM HR Analytics dataset** (1,470 employees, 35 features).

---

## 📊 Key Results

| Metric | Score |
|---|---|
| Model | CatBoost (Optuna-tuned) |
| Recall (attrition class) | **74%** |
| Precision (attrition class) | 43% |
| F1 Score | 54% |
| Accuracy | 80% |

> Recall is the primary metric — missing an employee who quits is far more costly than a false alarm.

---

## 🔍 Key Insights Discovered

1. **OverTime is the #1 attrition driver** — employees working overtime are 3× more likely to leave
2. **Sales department** has the highest attrition rate at ~20%
3. **New employees (0–2 years)** are the highest flight risk group
4. **No stock options** strongly predicts attrition — financial retention matters
5. **Low environment satisfaction** is a top 5 predictor of leaving

---

## 🏗️ Project Structure

```
hr-attrition-analyzer/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── 01_explore.ipynb        ← EDA & model development
│
├── src/
│   └── best_model.pkl          ← Saved CatBoost model
│
├── dashboard/
│   └── app.py                  ← Streamlit app (3 pages)
│
└── requirements.txt
```

---


## 📱 Dashboard Pages

### 🏠 Overview
High-level KPIs and visual insights — attrition by department, income distribution, tenure patterns.

### 🔍 Employee Risk Predictor
Input any employee's details and instantly get:
- Attrition probability (0–100%)
- Risk level (Low / Medium / High)
- SHAP explanation of top contributing factors

### 📊 Fleet Risk View
Full workforce table ranked by attrition risk — filterable by department, with risk level labels.

---

## 🧪 Modeling Journey

| Experiment | Recall | Notes |
|---|---|---|
| XGBoost (baseline) | 40.4% | Good starting point |
| LightGBM | 42.6% | Similar to XGBoost |
| CatBoost | 55.3% | Best — handles categoricals natively |
| CatBoost + SMOTE | 44.7% | Hurt performance — dataset too small |
| CatBoost + threshold tuning | 74.0% | Big recall jump, precision drop |
| **CatBoost + Optuna (final)** | **74.0%** | **Best balance — chosen model** |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas & numpy | Data processing |
| CatBoost | Primary ML model |
| XGBoost & LightGBM | Comparison models |
| SHAP | Model explainability |
| Optuna | Hyperparameter tuning |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| imbalanced-learn | SMOTE experiment |

---

## 💡 What I Learned

- **CatBoost outperforms on categorical-heavy datasets** — no encoding needed
- **SMOTE can hurt on small datasets** — class weights work better under 2K rows
- **Threshold tuning is underrated** — lowering from 0.5 to 0.3 boosted recall by 19%
- **SHAP explainability is what sells ML to business** — numbers alone don't convince decision-makers

---

## 📬 Contact

Built by Agti Nedjemeddine — Data Scientist specializing in predictive analytics and business intelligence.

- LinkedIn: [https://www.linkedin.com/in/nedjemeddine-agti-072662316/]
- GitHub: [https://github.com/Nedjmou07]
- Email: agtinedjmeddine2021.com

---

*⭐ If you found this project useful, give it a star!*

