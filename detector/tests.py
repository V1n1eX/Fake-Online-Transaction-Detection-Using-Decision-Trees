from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from detector.ml_model import _load_model_from_path, predict_transaction, train_random_forest
from detector.models import TransactionAnalysis


class PredictionFlowTests(TestCase):
	def test_pages_load(self):
		for path in ('/', '/predict/'):
			with self.subTest(path=path):
				self.assertEqual(self.client.get(path).status_code, 200)

	@patch('detector.ml_model.load_model', side_effect=FileNotFoundError)
	def test_prediction_form_reports_missing_trained_model(self, _load_model):
		response = self.client.post(
			'/predict/',
			{
				'step': '278',
				'type': 'CASH_IN',
				'amount': '330218.42',
				'nameOrig': 'C632336343',
				'oldbalanceOrg': '20866.00',
				'newbalanceOrig': '351084.42',
				'nameDest': 'C834976624',
				'oldbalanceDest': '452419.57',
				'newbalanceDest': '122201.15',
				'isFlaggedFraud': '0',
			},
		)

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'The screening model is unavailable')

	@patch(
		'detector.ml_model.predict_transaction',
		return_value={'prediction': 1, 'label': 'Fraudulent', 'risk_score': 0.72},
	)
	def test_prediction_is_saved_without_account_identifiers(self, _predict):
		response = self.client.post(
			'/predict/',
			{
				'step': '278',
				'type': 'TRANSFER',
				'amount': '5250.00',
				'nameOrig': 'origin-sensitive-id',
				'oldbalanceOrg': '6000',
				'newbalanceOrig': '750',
				'nameDest': 'destination-sensitive-id',
				'oldbalanceDest': '0',
				'newbalanceDest': '5250',
				'isFlaggedFraud': '0',
			},
		)

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'Potential fraud detected')
		self.assertEqual(TransactionAnalysis.objects.count(), 1)
		self.assertEqual(response.context['assessment']['category'], 'High')
		self.assertEqual(response.context['stats']['suspicious'], 1)
		record = TransactionAnalysis.objects.get()
		self.assertEqual(record.prediction, 'Fraudulent')
		self.assertFalse(hasattr(record, 'nameOrig'))

	def test_random_forest_trains_saves_and_predicts(self):
		with self.subTest('train, evaluate, save, and predict'):
			import tempfile
			from pathlib import Path

			with tempfile.TemporaryDirectory() as temporary_directory:
				model_path = Path(temporary_directory) / 'random_forest.pkl'
				dataframe = pd.DataFrame(
					{
						'amount': range(40),
						'channel': ['web', 'mobile'] * 20,
						'is_fraud': [int(amount > 20) for amount in range(40)],
					}
				)

				metrics = train_random_forest(dataframe, 'is_fraud', model_path)
				prediction = predict_transaction(
					{'amount': 35, 'channel': 'new-channel'}, model_path
				)

				self.assertTrue(model_path.is_file())
				self.assertEqual(set(metrics), {'accuracy', 'precision', 'recall', 'f1_score'})
				self.assertEqual(prediction['label'], 'Fraudulent')
				self.assertGreaterEqual(prediction['risk_score'], 0.0)
				self.assertLessEqual(prediction['risk_score'], 1.0)
				_load_model_from_path.cache_clear()

	def test_prediction_rejects_missing_model_features(self):
		with self.assertRaisesRegex(ValueError, 'missing model features'):
			predict_transaction({'amount': 100})
