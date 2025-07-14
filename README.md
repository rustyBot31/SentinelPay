# 🛡️ SentinelPay: Blockchain-Based Receipt Verification & Return Risk Prediction

SentinelPay is a full-stack application that enhances retail trust and fraud prevention by combining blockchain and machine learning. It enables secure storage and verification of purchase receipts using Ethereum smart contracts, while an AI model predicts the likelihood of a product being returned based on transaction data.<br>

### 🧠 Project concept by team - The Sentinels for Walmart Sparkathon 2025, aimed at transforming receipt validation and return risk analysis through innovative tech. <br>
### Authors - 
    - Swastik Bose
    - Yash Agarwal

## 🔧 Features
- Tamper-proof receipt storage on the blockchain via Solidity smart contracts.
- AI-powered return risk prediction using a trained ML model.
- User-friendly React frontend 
- Dual-panel interface for blockchain hash generation & AI analysis.
- Seamless wallet integration for transaction signing and verification.

## 💡 Use Case
Improves retail transparency and security by detecting suspicious return behavior and ensuring receipts are authentic and unaltered.

## 🧰 Tech Stack

### 🔗 Blockchain
- Solidity – Smart contract development
- Hardhat – Ethereum development environment and testing
- Ethers.js – Wallet interaction and contract calls

### 🧠 Machine Learning
- Python – Backend language for ML model
- Flask / FastAPI – Serving prediction endpoint ```(/predict-return-risk)```
- Scikit-learn / XGBoost – Return risk classification model

