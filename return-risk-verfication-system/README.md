# SentinelPay ML - Return Fraud Detection

A simple machine learning API for detecting fraudulent return requests in e-commerce.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   python sentinelpay_ml.py
   ```

3. **Test the API**
   ```bash
   python test_api.py
   ```

## API Usage

### Predict Return Risk
```bash
curl -X POST http://localhost:5000/predict-return-risk \
  -H "Content-Type: application/json" \
  -d '{"price": 1200, "days_since_purchase": 3, "return_history": 2, "geo_mismatch": 1}'
```

### Health Check
```bash
curl http://localhost:5000/health
```

## Input Parameters

- `price` (float): Item price in dollars
- `days_since_purchase` (float): Days between purchase and return request
- `return_history` (int): Number of previous returns by customer
- `geo_mismatch` (int): 1 if return location differs from purchase, 0 otherwise

## Response Format

```json
{
  "risk_score": 0.82,
  "flagged": true
}
```

## Files

- `sentinelpay_ml.py` - Main ML module and API
- `api_test.py` - API testing script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Demo Scenarios

The system will flag returns as high risk when:
- Expensive items (>$500) returned quickly (<7 days)
- Customers with multiple return history (>2 returns)
- Geographic mismatches between purchase and return locations

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: Price, days since purchase, return history, geographic mismatch
- **Training Data**: 5,000 synthetic samples with realistic fraud patterns
- **Performance**: ~80% accuracy on test data