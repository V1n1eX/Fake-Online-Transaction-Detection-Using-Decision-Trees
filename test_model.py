import sys
import os

sys.path.insert(0, '.')

import joblib
from detector.ml_model import predict_transaction

print('=== TESTING MODEL ===')

# Load the model
model = joblib.load('models/random_forest_model.pkl')
print('Model loaded successfully')
print('Model type:', type(model))
print('Model classes:', model.classes_)

# Test prediction with the first row of the dataset
payload = {
    'step': 278,
    'type': 'CASH_IN',
    'amount': 330218.42,
    'nameOrig': 'C632336343',
    'oldbalanceOrg': 20866.0,
    'newbalanceOrig': 351084.42,
    'nameDest': 'C834976624',
    'oldbalanceDest': 452419.57,
    'newbalanceDest': 122201.15,
    'isFlaggedFraud': 0
}
result = predict_transaction(payload)
print('\\nPrediction result:', result)
