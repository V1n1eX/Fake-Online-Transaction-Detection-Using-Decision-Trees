import json
import logging
import math
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .models import TransactionAnalysis

logger = logging.getLogger(__name__)

TRANSACTION_FIELDS = (
    'step',
    'type',
    'amount',
    'nameOrig',
    'oldbalanceOrg',
    'newbalanceOrig',
    'nameDest',
    'oldbalanceDest',
    'newbalanceDest',
    'isFlaggedFraud',
)
TRANSACTION_TYPES = {'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}


def _read_model_metadata():
    metadata_path = settings.BASE_DIR / 'results' / 'model_metrics.json'
    try:
        with metadata_path.open(encoding='utf-8') as metadata_file:
            return json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return {}


def _dashboard_context(values=None, error=None, assessment=None):
    from .ml_model import DEFAULT_MODEL_PATH

    model_metadata = _read_model_metadata()
    evaluation = model_metadata.get('evaluation', {})
    metrics = evaluation.get('metrics', {})
    recall = metrics.get('recall')
    if isinstance(recall, (int, float)) and math.isfinite(recall):
        recall_display = f'{recall * 100:.1f}%'
    else:
        recall_display = 'Unavailable'

    threshold = settings.FRAUD_REVIEW_THRESHOLD
    recent_analyses = list(TransactionAnalysis.objects.all()[:8])
    for record in recent_analyses:
        record.score_percent = float(record.model_score) * 100
        record.transaction_type_display = record.transaction_type.replace('_', ' ').title()
    return {
        'values': values or {},
        'error': error,
        'assessment': assessment,
        'model_available': DEFAULT_MODEL_PATH.is_file(),
        'threshold': threshold,
        'threshold_percent': f'{threshold * 100:.0f}%',
        'stats': {
            'total': TransactionAnalysis.objects.count(),
            'suspicious': TransactionAnalysis.objects.filter(prediction='Fraudulent').count(),
            'legitimate': TransactionAnalysis.objects.filter(prediction='Legitimate').count(),
            'recall': recall_display,
        },
        'recent_analyses': recent_analyses,
        'model_metadata': model_metadata,
        'features_count': len(TRANSACTION_FIELDS),
    }


def home(request):
    return predict_transaction(request)


@require_http_methods(['GET', 'POST'])
def predict_transaction(request):
    if request.method == 'GET':
        return render(request, 'detector/predict.html', _dashboard_context())

    values = {name: request.POST.get(name, '').strip() for name in TRANSACTION_FIELDS}
    payload = {}
    try:
        for name in TRANSACTION_FIELDS:
            value = values[name]
            if name == 'type':
                if value not in TRANSACTION_TYPES:
                    raise ValueError('Select a valid transaction type.')
                payload[name] = value
            elif name in {'nameOrig', 'nameDest'}:
                if not value:
                    raise ValueError('Complete both account identifiers.')
                payload[name] = value
            elif name == 'isFlaggedFraud':
                if value not in {'0', '1'}:
                    raise ValueError('Choose whether the source system flagged this transaction.')
                payload[name] = int(value)
            else:
                try:
                    numeric_value = float(value)
                except ValueError as exc:
                    raise ValueError('Enter a valid number for each amount and the transaction step.') from exc
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise ValueError('Amounts and the transaction step must be valid non-negative numbers.')
                if name == 'step':
                    if numeric_value < 1 or not numeric_value.is_integer():
                        raise ValueError('Enter a whole transaction step greater than zero.')
                    payload[name] = int(numeric_value)
                else:
                    payload[name] = numeric_value

        from .ml_model import predict_transaction as ml_predict

        prediction = ml_predict(payload)
        score = float(prediction['risk_score'])
        record = TransactionAnalysis.objects.create(
            transaction_type=payload['type'],
            amount=Decimal(str(payload['amount'])).quantize(Decimal('0.01')),
            prediction=prediction['label'],
            model_score=Decimal(str(score)).quantize(Decimal('0.0001')),
        )
        assessment = {
            'label': prediction['label'],
            'score': score,
            'score_percent': f'{score * 100:.1f}',
            'category': 'High' if score >= settings.FRAUD_REVIEW_THRESHOLD else 'Low',
            'transaction_type': payload['type'].replace('_', ' ').title(),
            'amount': record.amount,
            'created_at': record.created_at,
            'analysis_id': record.pk,
        }
    except FileNotFoundError:
        return render(
            request,
            'detector/predict.html',
            _dashboard_context(values, 'The screening model is unavailable. Check that the trained model file is installed.'),
        )
    except ValueError as exc:
        return render(request, 'detector/predict.html', _dashboard_context(values, str(exc)))
    except Exception:
        logger.exception('Transaction assessment failed')
        return render(
            request,
            'detector/predict.html',
            _dashboard_context(values, 'The transaction could not be assessed. Check the submitted values and try again.'),
        )

    return render(request, 'detector/predict.html', _dashboard_context(assessment=assessment))
