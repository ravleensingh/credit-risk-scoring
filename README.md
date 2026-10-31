# Intelligent Credit Risk Scoring & Agentic Lending Decision Support

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-purple.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end credit analytics project that combines classical machine learning with an agentic lending assistant. The system predicts repayment likelihood from structured borrower data, then extends that prediction into a lending assessment workflow with retrieved regulatory context and a structured decision report.

[Live Demo](https://credit-riskscoring.streamlit.app) | [GitHub Repository](https://github.com/ravleensingh/credit-risk-scoring)

## Overview

- Milestone 1: a Random Forest credit-risk model for repayment/default prediction.
- Milestone 2: a LangGraph-based lending advisor that adds risk reasoning, regulation retrieval, and report generation.
- Dataset size: 20,000 loan applications.
- Input features: 21 borrower and loan attributes.
- Target: `loan_paid_back` where `1 = paid back` and `0 = defaulted`.

## Final Status

- The ML pipeline has been refactored into a shared sklearn workflow used consistently by training scripts and both Streamlit pages.
- The advisor retrieval layer now supports semantic retrieval when the local embedding model is available and falls back to TF-IDF retrieval in offline or restricted environments.
- The UI now accepts more realistic custom inputs and warns when a prediction is outside the model's observed training ranges.
- Regression tests cover prediction validation, constraint handling, report parsing, retrieval fallback, and corrupt model-artifact recovery.

## Model Performance

### Validation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------:|
| **Random Forest** | **89.78%** | **0.8923** | **0.9919** | **0.9395** | **0.8738** |
| Logistic Regression | 88.67% | 0.9008 | 0.9647 | 0.9316 | 0.8854 |
| Decision Tree | 89.25% | 0.8948 | 0.9809 | 0.9359 | 0.8561 |

Random Forest remains the deployed model because it gives the strongest overall accuracy with very high recall on paid-back loans, even though Logistic Regression achieved the highest ROC-AUC.

### Random Forest Confusion Matrix

```text
                  Predicted
              Default  Paid Back
Actual Default    417       383
   Paid Back       26      3174
```

### Top Features by Permutation Importance

1. Employment Status — 69.7%
2. Debt-to-Income Ratio — 17.8%
3. Credit Score — 8.9%
4. Interest Rate — 1.0%
5. Grade/Subgrade — 0.7%

## Architecture

### Credit-Risk App

- Input collection through Streamlit forms.
- Shared sklearn preprocessing and prediction pipeline.
- Live repayment and default probability prediction.

### Agentic Lending Advisor

The advisor uses a three-step LangGraph workflow:

```text
START -> Risk Analyzer -> Regulation Retriever -> Report Generator -> END
```

- Risk Analyzer: summarizes borrower risk drivers using ML output and derived ratios.
- Regulation Retriever: retrieves relevant regulatory context from a local knowledge base.
- Report Generator: produces a four-section lending assessment report.

### Retrieval Strategy

- Primary path: FAISS + SentenceTransformers (`all-MiniLM-L6-v2`) when the model is locally available.
- Fallback path: TF-IDF lexical retrieval when the semantic model or its dependencies are unavailable.

## Streamlit Pages

- [app.py](app.py): primary credit-risk prediction interface.
- [pages/lending_advisor.py](pages/lending_advisor.py): agentic lending assessment interface.

## Repository Structure

```text
credit-risk-scoring/
├── agent/
│   ├── graph.py
│   ├── nodes.py
│   ├── rag.py
│   └── state.py
├── data/
│   └── regulations.txt
├── dataset/
│   └── original_dataset.csv
├── models/
│   └── model_params.py
├── notebooks/
│   └── eda.ipynb
├── pages/
│   └── lending_advisor.py
├── tests/
│   ├── test_input_options.py
│   ├── test_modeling.py
│   └── test_rag_and_reporting.py
├── utils/
│   ├── input_options.py
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── reporting.py
│   └── runtime.py
├── visualizations/
├── app.py
├── compare_models.py
├── project_architecture_diagram.py
├── requirements.txt
└── train_model.py
```

## Installation

```bash
git clone https://github.com/ravleensingh/credit-risk-scoring.git
cd credit-risk-scoring
python -m pip install -r requirements.txt
```

The app and scripts resolve datasets, visualizations, and generated model files relative to the repository, so commands can be run from the repository root or another working directory.

For the lending advisor page only, optionally add a Groq API key. Create a key at [Groq Console](https://console.groq.com/keys), then create a `.env` file in the repository root:

```bash
cp .env.example .env  # macOS/Linux
```

Set `GROQ_API_KEY` in `.env`. Alternatively, add the key to Streamlit Cloud's app secrets. Do not commit `.env` or `.streamlit/secrets.toml`; both are ignored by Git. The primary Random Forest prediction page does not need an API key.

To explore the notebook, install Jupyter separately if needed and open `notebooks/eda.ipynb`. It locates the dataset from the current directory or its parent directories, so it works when Jupyter is started from the repository root or `notebooks/`.

## Usage

### Train the Deployed Model

```bash
python train_model.py
```

### Compare Candidate Models

```bash
python compare_models.py
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Open the local app at `http://localhost:8501`. The lending advisor is available in the app's **Lending Advisor** page; its Groq-powered workflow needs a valid `GROQ_API_KEY`.

## Testing

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```

Core checks covered today, August 9, 2026:

- prediction input validation
- zero-interest installment handling
- training-range warning logic
- corrupt saved-model fallback
- retrieval fallback behavior
- deterministic regulation deduplication
- report section parsing

## Constraints and Known Limitations

- The deployed model was trained on historical data with limited observed ranges and categories. The UI allows broader custom input, but the app warns when it is extrapolating outside the training distribution.
- The dataset only contains 36-month and 60-month loan terms. Custom terms are supported in the UI, but predictions for other terms are out-of-distribution estimates.
- The lending advisor requires a valid `GROQ_API_KEY` for full end-to-end LLM execution.
- There is no browser-level automated integration test yet for the live Streamlit UI.

## Visual Outputs

Generated artifacts are saved in `visualizations/`, including:

- confusion matrices
- ROC curves
- model metric comparison
- final Random Forest feature importance

## Team

| Name | Role | Contributions |
|------|------|---------------|
| Ravleen Singh | Project Lead | Model development, agent architecture, deployment, integration |
| Anurag Pandey | Data Engineer | Data preprocessing, feature engineering, RAG knowledge base |
| Ansh Tomar | Data Analyst | EDA, visualization, documentation, regulatory research |
| Himanshu Chauhan | Frontend Developer | UI development, Streamlit pages, testing, UX |

## Institution

Newton School of Technology  
GenAI Capstone Project  
Reviewed and cleaned for final submission on August 9, 2026.
