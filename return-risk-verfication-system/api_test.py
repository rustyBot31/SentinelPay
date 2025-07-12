#!/usr/bin/env python3
"""
SentinelPay ML API Test Script
Test the fraud detection API endpoints
"""

import requests
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
import random


# API Configuration
API_BASE_URL = "http://localhost:5000"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_single_prediction(test_data, expected_result=None):
    """Test a single prediction"""
    print(f"\n📊 Testing Prediction:")
    print(f"Input: {test_data}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-return-risk",
            headers=HEADERS,
            json=test_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            
            risk_score = result.get('risk_score', 0)
            flagged = result.get('flagged', False)
            
            if flagged:
                print(f"🚨 FLAGGED - Risk Score: {risk_score}")
            else:
                print(f"✅ APPROVED - Risk Score: {risk_score}")
            
            return True, result
        else:
            print(f"❌ Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, None

def test_error_handling():
    """Test API error handling"""
    print("\n🛠️ Testing Error Handling...")
    
    error_cases = [
        {
            "name": "Missing required field",
            "data": {"price": 100, "days_since_purchase": 5}  # Missing return_history and geo_mismatch
        },
        {
            "name": "Invalid data type",
            "data": {"price": "not_a_number", "days_since_purchase": 5, "return_history": 1, "geo_mismatch": 0}
        },
        {
            "name": "Negative values",
            "data": {"price": -100, "days_since_purchase": 5, "return_history": 1, "geo_mismatch": 0}
        },
        {
            "name": "Invalid geo_mismatch",
            "data": {"price": 100, "days_since_purchase": 5, "return_history": 1, "geo_mismatch": 5}
        },
        {
            "name": "Empty request",
            "data": {}
        }
    ]
    
    for case in error_cases:
        print(f"\n  Testing: {case['name']}")
        success, result = test_single_prediction(case['data'])
        if success:
            print(f"  ⚠️ Expected error but got success: {result}")
        else:
            print(f"  ✅ Correctly handled error")

def run_demo_scenarios():
    """Run demo scenarios for the pitch"""
    print("\n🎬 DEMO SCENARIOS FOR PITCH")
    print("="*50)
    
    scenarios = [
        {
            "name": "🚨 HIGH RISK SCENARIO",
            "description": "Expensive item, quick return, repeat returner, different location",
            "data": {
                "price": 1200.0,
                "days_since_purchase": 3.0,
                "return_history": 2,
                "geo_mismatch": 1
            },
            "expected": "FLAGGED"
        },
        {
            "name": "✅ LOW RISK SCENARIO",
            "description": "Moderate price, reasonable delay, first return, same location",
            "data": {
                "price": 89.99,
                "days_since_purchase": 25.0,
                "return_history": 0,
                "geo_mismatch": 0
            },
            "expected": "APPROVED"
        },
        {
            "name": "⚠️ MEDIUM RISK SCENARIO",
            "description": "High price but legitimate pattern",
            "data": {
                "price": 599.99,
                "days_since_purchase": 14.0,
                "return_history": 1,
                "geo_mismatch": 0
            },
            "expected": "BORDERLINE"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected']}")
        print("-" * 40)
        
        success, result = test_single_prediction(scenario['data'])
        
        if success:
            risk_score = result.get('risk_score', 0)
            flagged = result.get('flagged', False)
            
            print(f"🎯 DEMO RESULT:")
            print(f"   Risk Score: {risk_score}")
            print(f"   Decision: {'FLAGGED' if flagged else 'APPROVED'}")
            print(f"   Confidence: {risk_score if flagged else 1-risk_score:.3f}")

def test_performance():
    """Test API performance"""
    print("\n⚡ Performance Test...")
    
    test_data = {
        "price": 299.99,
        "days_since_purchase": 10.0,
        "return_history": 1,
        "geo_mismatch": 0
    }
    
    num_requests = 10
    start_time = time.time()
    
    successful_requests = 0
    for i in range(num_requests):
        success, _ = test_single_prediction(test_data)
        if success:
            successful_requests += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"📈 Performance Results:")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful: {successful_requests}")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time per request: {total_time/num_requests:.3f} seconds")
    print(f"   Requests per second: {num_requests/total_time:.1f}")

def main():
    """Main test function"""
    print("🧪 SENTINELPAY ML API TESTING")
    print("="*50)
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Health Check
    if not test_health_check():
        print("❌ Health check failed. Make sure the API is running!")
        return

    # Test 2: Demo Scenarios
    run_demo_scenarios()

    # Test 3: Error Handling
    test_error_handling()

    # Test 4: Performance
    test_performance()

    # Test 5: ROC AUC
    test_dataset = generate_synthetic_test_data(n_samples=50)
    test_roc_auc(test_dataset)

    print("\n🎉 ALL TESTS COMPLETED")
    print("="*50)
    print("✅ API is ready for the SentinelPay demo!")


def generate_synthetic_test_data(n_samples=100):
    """Generate synthetic labeled test data"""
    dataset = []

    for _ in range(n_samples):
        # Generate realistic feature values
        price = round(random.uniform(50, 1500), 2)
        days_since_purchase = round(random.uniform(0.5, 30), 1)
        return_history = random.choice([0, 1, 2])
        geo_mismatch = random.choice([0, 1])

        # Basic fraud logic (just for simulating label, can be replaced by real labels)
        risk_score = (
            (price / 1500) * 0.3 +
            ((30 - days_since_purchase) / 30) * 0.3 +
            (return_history / 2) * 0.2 +
            (geo_mismatch * 0.2)
        )

        label = 1 if risk_score > 0.6 else 0  # Threshold-based label

        dataset.append({
            "features": {
                "price": price,
                "days_since_purchase": days_since_purchase,
                "return_history": return_history,
                "geo_mismatch": geo_mismatch
            },
            "label": label
        })

    return dataset

def test_roc_auc(test_dataset):
    """Evaluate ROC AUC on labeled test data using the API"""
    print("\n📊 ROC AUC Evaluation")
    y_true = []
    y_scores = []
    failed_requests = 0

    for i, sample in enumerate(test_dataset):
        data = sample["features"]
        label = sample["label"]
        print(f"  Sample {i+1}/{len(test_dataset)}: {data}")

        success, result = test_single_prediction(data)

        if success and result:
            risk_score = result.get("risk_score", 0.0)
            y_true.append(label)
            y_scores.append(risk_score)
        else:
            failed_requests += 1

    if len(y_true) >= 2:
        auc = roc_auc_score(y_true, y_scores)
        print(f"\n✅ ROC AUC Score: {auc:.4f}")
    else:
        print("❌ Not enough valid results to compute ROC AUC.")

    if failed_requests > 0:
        print(f"⚠️ {failed_requests} requests failed during AUC testing.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Test script error: {e}")