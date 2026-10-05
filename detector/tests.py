from django.test import TestCase
from unittest.mock import patch


class PredictionFlowTests(TestCase):
	def test_pages_load(self):
		for path in ('/', '/predict/', '/about/', '/result/'):
			with self.subTest(path=path):
				self.assertEqual(self.client.get(path).status_code, 200)

	@patch('detector.ml_model.load_model', side_effect=FileNotFoundError)
	def test_prediction_form_reports_missing_trained_model(self, _load_model):
		response = self.client.post(
			'/predict/',
			{
				'amount': '125.50',
				'transaction_hour': '22',
				'merchant_risk': '0.72',
				'device_risk': '0.34',
				'location_risk': '0.68',
				'customer_tenure_days': '215',
				'avg_transaction_amount': '90.00',
				'ip_risk': '0.58',
			},
		)

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'Random Forest model file is not available yet')
