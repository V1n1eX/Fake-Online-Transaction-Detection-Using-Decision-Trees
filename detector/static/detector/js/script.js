document.addEventListener('DOMContentLoaded', () => {
    const sampleButton = document.getElementById('load-sample');

    if (!sampleButton) {
        return;
    }

    sampleButton.addEventListener('click', () => {
        const sampleValues = {
            amount: '245.80',
            transaction_hour: '23',
            merchant_risk: '0.91',
            device_risk: '0.56',
            location_risk: '0.83',
            customer_tenure_days: '120',
            avg_transaction_amount: '180.40',
            ip_risk: '0.76'
        };

        Object.entries(sampleValues).forEach(([field, value]) => {
            const element = document.getElementById(field);
            if (element) {
                element.value = value;
            }
        });
    });
});
