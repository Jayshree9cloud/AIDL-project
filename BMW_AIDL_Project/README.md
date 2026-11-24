# BMW – Innovation in Automotive Technology
## AIDL Course Project: Predictive Maintenance Prototype

This project follows the scope and methods in the supplied AIDL report.

### What it does
1. Downloads the UCI AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic industrial records).
2. Inspects the data and target distribution.
3. Removes identifiers (`UDI`, `Product ID`) and failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to avoid target leakage.
4. Encodes the `Type` category, imputes missing values, and scales numerical features.
5. Uses a stratified 80:20 train/test split.
6. Trains and evaluates:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Artificial Neural Network (ANN)
7. Reports accuracy, precision, recall, F1-score and confusion matrices.
8. Saves plots, a model comparison CSV, model files and one illustrative ANN prediction.

### Important dataset limitation
The AI4I dataset is synthetic industrial-machine data. It is **not BMW vehicle data**, BMW factory data, or BMW service-centre data. It is used only as a public educational proxy to demonstrate predictive-maintenance methods relevant to modern automotive and industrial environments. The model is not a validated vehicle safety system and must not be used for real maintenance or road-safety decisions.

## Run on Windows, macOS or Linux

### 1. Install Python
Use Python 3.10 or 3.11 (recommended for TensorFlow compatibility).

### 2. Open a terminal in this project folder

### 3. Create and activate a virtual environment

**Windows**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the project
```bash
python main.py
```

The first run requires an internet connection to fetch the dataset from UCI. TensorFlow installation may take a few minutes.

## Google Colab
1. Upload `BMW_AIDL_Predictive_Maintenance.ipynb` to Google Colab.
2. Select Runtime → Run all.
3. The notebook installs its dependencies and fetches the dataset.
4. Download the generated `outputs` folder from the Colab Files panel if needed.

## Output files
The program creates:
- `outputs/dataset_summary.json`
- `outputs/model_comparison.csv`
- Confusion-matrix PNG and JSON files for each model
- `outputs/ann_training_loss.png`
- `outputs/ann_training_accuracy.png`
- `outputs/model_performance_comparison.png`
- `outputs/illustrative_prediction.json`
- `models/` containing the trained ML models, ANN and fitted preprocessor

Actual metrics are generated when you run the code. No accuracy values are fabricated in advance.

## Project algorithms
- Logistic Regression: baseline binary classifier.
- Decision Tree: learns feature-threshold decision rules.
- Random Forest: combines multiple decision trees.
- ANN: Dense(32, ReLU) → Dropout(0.20) → Dense(16, ReLU) → Dense(1, Sigmoid), trained with Adam and binary cross-entropy.

## Troubleshooting
- If TensorFlow cannot be installed, check that you are using a supported 64-bit Python version (Python 3.10 or 3.11 is recommended).
- If the dataset download fails, check your internet connection and try again.
- If installation is slow, allow several minutes for TensorFlow and its dependencies.
