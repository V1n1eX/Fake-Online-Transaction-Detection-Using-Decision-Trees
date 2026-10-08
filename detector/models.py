from django.db import models


class TransactionAnalysis(models.Model):
    transaction_type = models.CharField(max_length=16)
    amount = models.DecimalField(max_digits=24, decimal_places=2)
    prediction = models.CharField(max_length=16)
    model_score = models.DecimalField(max_digits=6, decimal_places=4)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'transaction analyses'

    def __str__(self):
        return f'{self.transaction_type} {self.amount} — {self.prediction}'
