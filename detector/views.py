from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods


def home(request):
    return render(request, 'detector/home.html')


def about(request):
    return render(request, 'detector/about.html')


@require_http_methods(['GET', 'POST'])
def predict_transaction(request):
    feature_names = [
        'amount',
        'transaction_hour',
        'merchant_risk',
        'device_risk',
        'location_risk',
        'customer_tenure_days',
        'avg_transaction_amount',
        'ip_risk',
    ]

    if request.method == 'POST':
        payload = {}
        for name in feature_names:
            try:
                payload[name] = float(request.POST.get(name, 0.0))
            except ValueError:
                payload[name] = 0.0

        try:
            from .ml_model import predict_transaction as ml_predict

            prediction = ml_predict(payload)
            request.session['last_prediction'] = prediction['label']
            request.session['last_prediction_value'] = prediction['prediction']
            request.session['last_risk_score'] = prediction['risk_score']
            request.session['last_payload'] = payload
            return redirect('result')
        except FileNotFoundError:
            return render(
                request,
                'detector/predict.html',
                {
                    'error': 'The Random Forest model file is not available yet. The application is ready to accept the trained model once it is saved.'
                },
            )

    return render(request, 'detector/predict.html')


def prediction_result(request):
    prediction = request.session.get('last_prediction', 'No prediction yet.')
    payload = request.session.get('last_payload', {})
    risk_score = request.session.get('last_risk_score', 0.0)
    return render(
        request,
        'detector/result.html',
        {
            'prediction': prediction,
            'risk_score': risk_score,
            'payload': payload,
        },
    )
