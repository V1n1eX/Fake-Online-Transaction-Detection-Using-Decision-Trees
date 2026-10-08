# Fake Online Transaction Detection Using Random Forest

This Django web application classifies PaySim transactions with a trained Random Forest pipeline. It uses the dataset in `dataset/onlinefraud.csv` and the model artifact at `../models/random_forest_model.pkl` when the app-local `models/random_forest_model.pkl` is absent. The model is loaded once and cached; it is not retrained during web requests.

## Project information
- Student: Vincent Kondowe
- Registration Number: 23311351014
- Project Serial Number: 7

## Technology stack
- Python
- Django
- scikit-learn
- pandas
- NumPy
- matplotlib
- SQLite

## Run locally on Windows
Run these commands from the repository root (the folder containing `.venv` and `fake_transaction_detector`):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\fake_transaction_detector\requirements.txt
Set-Location .\fake_transaction_detector
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/ in a browser. If PowerShell blocks environment activation, run the commands with `..\.venv\Scripts\python.exe` from the app directory instead.

## Train the model
From `fake_transaction_detector/`, run:

```powershell
python train_model.py
```

The script reads `dataset/onlinefraud.csv`, reports held-out accuracy, precision, recall, and F1, then saves the fitted pipeline to `fake_transaction_detector/models/random_forest_model.pkl`.

The dashboard stores completed analyses in SQLite so its totals and recent history reflect actual submissions. It records only transaction type, amount, model label, model score, and timestamp; source and destination account identifiers are not stored. The held-out metrics are written to `results/model_metrics.json` when training completes.

`FRAUD_REVIEW_THRESHOLD` controls the dashboard's high-risk review category. It defaults to `0.5` and accepts a value from `0` through `1`. The model score is shown as an uncalibrated classifier score, separate from the classifier's prediction label.

## Latest held-out evaluation
The current training run used a stratified 80/20 split with random seed 42:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.99896 |
| Precision | 1.00000 |
| Recall | 0.23810 |
| F1 score | 0.38462 |

The dataset is highly imbalanced (76,819 legitimate rows and 107 fraud rows with labels). The high accuracy does not mean the model catches most fraud: this run detected about 24% of fraud in the held-out split. Treat the app as an academic demonstration, and improve fraud recall before using it for decisions.

## Project structure
- `detector/` contains the Django application and ML integration layer.
- `models/` is the app-local destination for the trained Random Forest pickle file.
- `dataset/` keeps raw or processed transaction datasets.
- `notebooks/` is reserved for experimentation and model analysis.
- `results/` stores evaluation summaries and visual outputs.

The prediction form uses the transaction fields expected by the saved model. The origin and destination account identifiers come from the dataset and are model inputs; they should be handled according to the project's privacy requirements in any real deployment.
