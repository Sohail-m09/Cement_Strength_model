## Cement_Strength_model
This project provides a professional, modular machine learning pipeline to predict the compressive strength of cement based on its composition and age. It uses XGBoost to achieve high-accuracy results and follows modern MLOps practices using DVC for data versioning.

## Key Highlights 
Best Model: XGBoost Regressor (Achieved R2 Score: 0.91).

Modular Architecture: Clean, decoupled code split into Ingestion, Preprocessing, Build, and Evaluation modules.

Pipeline Orchestration: Uses DVC (Data Version Control) to manage data lineage and reproducible workflows.

Efficient Environment: Managed using uv for lightning-fast dependency installation.

## Tech Stack and Dependencies

Core Libraries:
Data Handling: pandas, numpy

Machine Learning: scikit-learn, xgboost, statsmodels

Visualization: matplotlib, seaborn

MLOps & Tracking: dvc, mlflow

Utilities: loguru (Advanced Logging), jinja2, ipykernel

Further Usage (Optional):
Flask or FastAPI (for model deployment), evidently (for data drift monitoring), Pytest (for unit testing).

## Project Structure

├── data/               # Versioned data (managed by DVC)
├── src/                # Modular Source Code
│   ├── data_ingestion.py     # Download/Load raw data
│   ├── data_preprocessing.py # Cleaning & Feature Engineering
│   ├── model_build.py        # Model initialization & dictionary
│   └── model_evaluation.py   # Training, Metrics & Pickling
├── main.py             # Entry point to run the whole pipeline
├── dvc.yaml            # DVC Pipeline configuration
└── requirements.txt    # Project dependencies

## Getting Started 

1. Setup Environment (Using uv)
uv is a fast replacement for pip. If you don't have it, install it via pip install uv.

Bash
Create a virtual environment
uv venv

Activate environment
On Windows:
.venv\Scripts\activate
On Mac/Linux:
source .venv/bin/activate

Install dependencies
uv pip install -r requirements.txt
2. Initialize DVC
Bash
dvc init

## How to run project 

You can run the entire pipeline in two ways:

** Option A: The Modular Way (Recommended)
Run the DVC pipeline to ensure all dependencies and data versions are tracked.

Bash
dvc repro
This will automatically run ingestion → preprocessing → build → evaluation.

** Option B: The Manual Way
Run the main script directly:

Bash
python main.py

## Insgihts 
The analysis confirms that Cement Quantity and Age are the most significant predictors of concrete strength. Using the XGBoost model, we can predict the 28-day strength with 91% accuracy, allowing for better material planning and reduced waiting times in construction.

CI entry point: main.py