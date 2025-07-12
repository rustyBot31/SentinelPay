#!/usr/bin/env python3
"""
SentinelPay ML Module - Improved Version
High-performance fraud detection with realistic synthetic data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. IMPROVED SYNTHETIC DATA GENERATION
# ================================

def generate_realistic_data(n_samples=10000, seed=42):
    """Generate realistic synthetic return data with clear fraud patterns"""
    np.random.seed(seed)
    
    data = []
    for i in range(n_samples):
        # Generate customer profile
        customer_tier = np.random.choice(['bronze', 'silver', 'gold'], p=[0.6, 0.3, 0.1])
        account_age_days = np.random.exponential(scale=365) + 30  # At least 30 days old
        
        # Generate transaction features
        if customer_tier == 'bronze':
            price = np.random.lognormal(mean=3.5, sigma=0.8)  # $30-$300 range
        elif customer_tier == 'silver':
            price = np.random.lognormal(mean=4.2, sigma=0.9)  # $70-$800 range
        else:  # gold
            price = np.random.lognormal(mean=5.0, sigma=1.0)  # $150-$2000 range
        
        # Time-based features
        days_since_purchase = np.random.exponential(scale=15)
        hour_of_return = np.random.choice(range(24))
        is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Behavioral features
        return_history = np.random.poisson(lam=0.8)
        returns_last_30_days = np.random.poisson(lam=0.3)
        
        # Geographic and shipping
        geo_mismatch = np.random.choice([0, 1], p=[0.85, 0.15])
        shipping_speed = np.random.choice(['standard', 'express', 'overnight'], p=[0.6, 0.3, 0.1])
        
        # Payment features
        payment_method = np.random.choice(['credit', 'debit', 'digital'], p=[0.5, 0.3, 0.2])
        
        # Product features
        category = np.random.choice(['electronics', 'clothing', 'home', 'books'], p=[0.3, 0.4, 0.2, 0.1])
        
        # REALISTIC FRAUD PROBABILITY CALCULATION
        fraud_prob = 0.02  # Base 2% fraud rate
        
        # Strong fraud indicators (multiplicative)
        if days_since_purchase < 2:
            fraud_prob *= 8.0  # Very quick returns are highly suspicious
        elif days_since_purchase < 7:
            fraud_prob *= 3.0
        elif days_since_purchase > 90:
            fraud_prob *= 2.0  # Very late returns also suspicious
        
        if return_history > 3:
            fraud_prob *= 6.0  # Heavy returners
        elif return_history > 1:
            fraud_prob *= 2.5
        
        if returns_last_30_days > 2:
            fraud_prob *= 4.0  # Recent return spree
        
        if geo_mismatch:
            fraud_prob *= 3.5  # Geographic inconsistency
        
        if price > 1000:
            fraud_prob *= 2.5  # High-value items
        elif price > 500:
            fraud_prob *= 1.5
        
        # Time-based patterns
        if 2 <= hour_of_return <= 6:  # Late night/early morning
            fraud_prob *= 2.0
        
        if account_age_days < 30:
            fraud_prob *= 4.0  # New accounts
        elif account_age_days < 90:
            fraud_prob *= 2.0
        
        # Category-specific risks
        if category == 'electronics':
            fraud_prob *= 1.8  # Electronics have higher fraud rates
        
        # Payment method risks
        if payment_method == 'digital':
            fraud_prob *= 1.5  # Digital payments can be riskier
        
        # Protective factors (reduce fraud probability)
        if customer_tier == 'gold':
            fraud_prob *= 0.3  # Gold customers are more trustworthy
        elif customer_tier == 'silver':
            fraud_prob *= 0.6
        
        if shipping_speed == 'overnight':
            fraud_prob *= 0.7  # Overnight shipping suggests legitimate urgency
        
        # Cap probability at 95%
        fraud_prob = min(fraud_prob, 0.95)
        
        # Generate label
        is_fraud = np.random.random() < fraud_prob
        
        # Create derived features
        price_return_ratio = price / (return_history + 1)
        return_velocity = returns_last_30_days / min(account_age_days / 30, 1)
        
        data.append({
            'price': round(price, 2),
            'days_since_purchase': round(days_since_purchase, 1),
            'return_history': return_history,
            'returns_last_30_days': returns_last_30_days,
            'geo_mismatch': geo_mismatch,
            'hour_of_return': hour_of_return,
            'is_weekend': is_weekend,
            'account_age_days': round(account_age_days, 1),
            'customer_tier_bronze': 1 if customer_tier == 'bronze' else 0,
            'customer_tier_silver': 1 if customer_tier == 'silver' else 0,
            'customer_tier_gold': 1 if customer_tier == 'gold' else 0,
            'shipping_express': 1 if shipping_speed == 'express' else 0,
            'shipping_overnight': 1 if shipping_speed == 'overnight' else 0,
            'payment_credit': 1 if payment_method == 'credit' else 0,
            'payment_digital': 1 if payment_method == 'digital' else 0,
            'category_electronics': 1 if category == 'electronics' else 0,
            'category_clothing': 1 if category == 'clothing' else 0,
            'category_home': 1 if category == 'home' else 0,
            'price_return_ratio': round(price_return_ratio, 2),
            'return_velocity': round(return_velocity, 3),
            'is_fraud': int(is_fraud)
        })
    
    return pd.DataFrame(data)

# ================================
# 2. ENHANCED ML MODEL
# ================================

class AdvancedFraudPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.threshold = 0.5
        
    def train(self, df):
        """Train the advanced fraud detection model"""
        # Prepare features (exclude target)
        feature_cols = [col for col in df.columns if col != 'is_fraud']
        X = df[feature_cols]
        y = df['is_fraud']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model Performance:")
        print(f"ROC AUC: {auc_score:.3f}")
        print(f"Fraud rate: {y.mean():.1%}")
        print(f"Training samples: {len(X_train)}")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
        print(f"  False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")
        
        return self
    
    def predict_risk(self, features):
        """Predict fraud risk with comprehensive feature validation"""
        try:
            # Define required basic features
            required_fields = ['price', 'days_since_purchase', 'return_history', 'geo_mismatch']
            
            # Check for required fields
            for field in required_fields:
                if field not in features:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create feature vector with defaults
            data = {
                'price': float(features['price']),
                'days_since_purchase': float(features['days_since_purchase']),
                'return_history': int(features['return_history']),
                'returns_last_30_days': int(features.get('returns_last_30_days', 0)),
                'geo_mismatch': int(features['geo_mismatch']),
                'hour_of_return': int(features.get('hour_of_return', 12)),
                'is_weekend': int(features.get('is_weekend', 0)),
                'account_age_days': float(features.get('account_age_days', 365)),
                'customer_tier_bronze': int(features.get('customer_tier_bronze', 1)),
                'customer_tier_silver': int(features.get('customer_tier_silver', 0)),
                'customer_tier_gold': int(features.get('customer_tier_gold', 0)),
                'shipping_express': int(features.get('shipping_express', 0)),
                'shipping_overnight': int(features.get('shipping_overnight', 0)),
                'payment_credit': int(features.get('payment_credit', 1)),
                'payment_digital': int(features.get('payment_digital', 0)),
                'category_electronics': int(features.get('category_electronics', 0)),
                'category_clothing': int(features.get('category_clothing', 1)),
                'category_home': int(features.get('category_home', 0)),
            }
            
            # Calculate derived features
            data['price_return_ratio'] = data['price'] / (data['return_history'] + 1)
            data['return_velocity'] = data['returns_last_30_days'] / max(data['account_age_days'] / 30, 1)
            
            # Validate ranges
            if data['price'] < 0 or data['days_since_purchase'] < 0:
                raise ValueError("Price and days_since_purchase must be non-negative")
            
            if data['geo_mismatch'] not in [0, 1]:
                raise ValueError("geo_mismatch must be 0 or 1")
            
            # Prepare for prediction
            X = pd.DataFrame([data])
            X = X[self.feature_names]  # Ensure correct order
            X_scaled = self.scaler.transform(X)
            
            # Get prediction
            risk_score = self.model.predict_proba(X_scaled)[0, 1]
            flagged = risk_score >= self.threshold
            
            # Risk level categorization
            if risk_score >= 0.8:
                risk_level = "VERY HIGH"
            elif risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            elif risk_score >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "VERY LOW"
            
            return {
                'risk_score': round(float(risk_score), 3),
                'risk_level': risk_level,
                'flagged': bool(flagged),
                'recommendation': 'BLOCK' if flagged else 'APPROVE'
            }
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.threshold = data['threshold']
        print(f"Model loaded from {filepath}")

# ================================
# 3. FLASK API
# ================================

app = Flask(__name__)
CORS(app)
predictor = AdvancedFraudPredictor()

@app.route('/predict-return-risk', methods=['POST'])
def predict_return_risk():
    """Enhanced API endpoint for fraud prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        result = predictor.predict_risk(data)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'SentinelPay ML Enhanced'})

# ================================
# 4. ENHANCED DEMO EXAMPLES
# ================================

def run_enhanced_demo():
    """Run comprehensive demo examples"""
    print("\n" + "="*70)
    print("SENTINELPAY ENHANCED ML DEMO")
    print("="*70)
    
    examples = [
        {
            'name': 'VERY HIGH RISK - Quick return, new account, high-value electronics',
            'data': {
                'price': 1599.99,
                'days_since_purchase': 1,
                'return_history': 0,
                'returns_last_30_days': 0,
                'geo_mismatch': 1,
                'hour_of_return': 3,
                'account_age_days': 15,
                'customer_tier_bronze': 1,
                'customer_tier_silver': 0,
                'customer_tier_gold': 0,
                'category_electronics': 1,
                'category_clothing': 0,
                'payment_digital': 1,
                'payment_credit': 0
            }
        },
        {
            'name': 'HIGH RISK - Repeat returner with geographic mismatch',
            'data': {
                'price': 299.99,
                'days_since_purchase': 2,
                'return_history': 4,
                'returns_last_30_days': 3,
                'geo_mismatch': 1,
                'hour_of_return': 14,
                'account_age_days': 180,
                'customer_tier_bronze': 1,
                'customer_tier_silver': 0,
                'customer_tier_gold': 0,
                'category_clothing': 1,
                'category_electronics': 0
            }
        },
        {
            'name': 'LOW RISK - Gold customer, reasonable timing',
            'data': {
                'price': 199.99,
                'days_since_purchase': 14,
                'return_history': 1,
                'returns_last_30_days': 0,
                'geo_mismatch': 0,
                'hour_of_return': 10,
                'account_age_days': 720,
                'customer_tier_bronze': 0,
                'customer_tier_silver': 0,
                'customer_tier_gold': 1,
                'category_clothing': 1,
                'category_electronics': 0,
                'shipping_overnight': 1
            }
        },
        {
            'name': 'MEDIUM RISK - Mixed signals',
            'data': {
                'price': 449.99,
                'days_since_purchase': 8,
                'return_history': 2,
                'returns_last_30_days': 1,
                'geo_mismatch': 0,
                'hour_of_return': 16,
                'account_age_days': 365,
                'customer_tier_bronze': 0,
                'customer_tier_silver': 1,
                'customer_tier_gold': 0,
                'category_electronics': 1,
                'category_clothing': 0
            }
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"Key features: Price=${example['data']['price']}, "
              f"Days={example['data']['days_since_purchase']}, "
              f"Return History={example['data']['return_history']}")
        
        try:
            result = predictor.predict_risk(example['data'])
            status_emoji = "🚨" if result['flagged'] else "✅"
            print(f"Result: {status_emoji} {result['recommendation']} "
                  f"(Risk: {result['risk_level']}, Score: {result['risk_score']})")
        except Exception as e:
            print(f"Error: {e}")

# ================================
# 5. MAIN EXECUTION
# ================================

def main():
    """Main function with enhanced training and validation"""
    print("SentinelPay Enhanced ML Module - Starting...")
    
    # Generate realistic training data
    print("\n1. Generating realistic synthetic training data...")
    df = generate_realistic_data(n_samples=10000)
    print(f"Generated {len(df)} samples")
    print(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    print(f"Feature count: {len(df.columns)-1}")
    
    # Train enhanced model
    print("\n2. Training enhanced fraud detection model...")
    predictor.train(df)
    
    # Save model
    print("\n3. Saving trained model...")
    predictor.save_model('sentinelpay_enhanced_model.pkl')
    
    # Run comprehensive demo
    print("\n4. Running enhanced demo examples...")
    run_enhanced_demo()
    
    print("\n" + "="*70)
    print("🚀 SENTINELPAY ENHANCED ML API READY")
    print("="*70)
    print("API Endpoints:")
    print("  POST /predict-return-risk  - Enhanced prediction endpoint")
    print("  GET  /health              - Health check")
    print("\nMinimal request example:")
    print('curl -X POST http://localhost:5000/predict-return-risk \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"price": 1599.99, "days_since_purchase": 1, "return_history": 0, "geo_mismatch": 1}\'')
    print("\nExpected response:")
    print('{"risk_score": 0.847, "risk_level": "VERY HIGH", "flagged": true, "recommendation": "BLOCK"}')

if __name__ == "__main__":
    try:
        main()
        print("\n🌟 Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")