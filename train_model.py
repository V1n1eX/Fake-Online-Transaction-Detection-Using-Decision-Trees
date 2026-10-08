import os
import sys
import json
from pathlib import Path
from pathlib import Path

sys.path.insert(0, '.')

import pandas as pd
from detector.ml_model import train_random_forest

# Resolve dataset path relative to this script (app directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'onlinefraud.csv')

print('=== LOADING DATASET ===')
print('Dataset path:', DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Target distribution (isFraud):')
print(df['isFraud'].value_counts(dropna=False))
print()

print('=== TRAINING ===')
metrics = train_random_forest(
    df,
    'isFraud',
    model_path='models/random_forest_model.pkl'
)
print('Metrics:', json.dumps(metrics, indent=2))
print()

labelled = df.dropna(subset=['isFraud'])
metadata = {
    'model_name': 'Random Forest',
    'dataset_file': os.path.basename(DATASET_PATH),
    'dataset': {
        'rows': int(len(df)),
        'labelled_rows': int(len(labelled)),
        'fraud_rows': int((labelled['isFraud'] == 1).sum()),
        'legitimate_rows': int((labelled['isFraud'] == 0).sum()),
    },
    'evaluation': {
        'method': 'Stratified train/test split',
        'test_fraction': 0.2,
        'random_state': 42,
        'metrics': metrics,
    },
    'feature_count': int(len(df.columns) - 1),
    'score_calibrated': False,
}
metrics_path = Path(BASE_DIR) / 'results' / 'model_metrics.json'
metrics_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print('Evaluation metadata saved to:', metrics_path)

labelled_rows = df.dropna(subset=['isFraud'])
metrics_path = Path(BASE_DIR) / 'results' / 'model_metrics.json'
metrics_path.parent.mkdir(parents=True, exist_ok=True)
model_metadata = {
    'model_name': 'Random Forest',
    'dataset_file': os.path.basename(DATASET_PATH),
    'dataset': {
        'rows': int(len(df)),
        'labelled_rows': int(len(labelled_rows)),
        'fraud_rows': int((labelled_rows['isFraud'] == 1).sum()),
        'legitimate_rows': int((labelled_rows['isFraud'] == 0).sum()),
    },
    'evaluation': {
        'method': 'Stratified train/test split',
        'test_fraction': 0.2,
        'random_state': 42,
        'metrics': metrics,
    },
    'feature_count': int(len(df.columns) - 1),
    'score_calibrated': False,
}
metrics_path.write_text(json.dumps(model_metadata, indent=2) + '\n', encoding='utf-8')
print('Evaluation metadata saved to:', metrics_path)

print('Model saved to:', os.path.abspath('models/random_forest_model.pkl'))
print('File size (bytes):', os.path.getsize('models/random_forest_model.pkl'))

