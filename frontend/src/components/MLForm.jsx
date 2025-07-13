import React, { useState } from "react";
import { verifyReceiptHash } from "../utils/blockchain";
import { predictReturnRisk } from "../services/mlApi";
import "./MLForm.css";

const MLForm = () => {
  const [formData, setFormData] = useState({
    receipt_text: "",
    price: "",
    days_since_purchase: "",
    return_history: "",
    returns_last_30_days: "",
    geo_mismatch: false,
    hour_of_return: "",
    is_weekend: false,
    account_age_days: "",
    customer_tier: "bronze",
    shipping_speed: "standard",
    payment_method: "credit",
    category: "clothing",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const buildMLPayload = () => {
    const {
      price,
      days_since_purchase,
      return_history,
      returns_last_30_days,
      geo_mismatch,
      hour_of_return,
      is_weekend,
      account_age_days,
      customer_tier,
      shipping_speed,
      payment_method,
      category,
    } = formData;

    return {
      price: parseFloat(price),
      days_since_purchase: parseFloat(days_since_purchase),
      return_history: parseInt(return_history),
      returns_last_30_days: parseInt(returns_last_30_days),
      geo_mismatch: geo_mismatch ? 1 : 0,
      hour_of_return: parseInt(hour_of_return),
      is_weekend: is_weekend ? 1 : 0,
      account_age_days: parseFloat(account_age_days),
      customer_tier_bronze: customer_tier === "bronze" ? 1 : 0,
      customer_tier_silver: customer_tier === "silver" ? 1 : 0,
      customer_tier_gold: customer_tier === "gold" ? 1 : 0,
      shipping_express: shipping_speed === "express" ? 1 : 0,
      shipping_overnight: shipping_speed === "overnight" ? 1 : 0,
      payment_credit: payment_method === "credit" ? 1 : 0,
      payment_digital: payment_method === "digital" ? 1 : 0,
      category_electronics: category === "electronics" ? 1 : 0,
      category_clothing: category === "clothing" ? 1 : 0,
      category_home: category === "home" ? 1 : 0,
    };
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      const { hash, isValid } = await verifyReceiptHash(formData.receipt_text);
      if (!isValid) {
        setError("❌ Receipt hash not found on blockchain.");
        return;
      }

      const payload = buildMLPayload();
      const mlResult = await predictReturnRisk(payload);
      setResult({ ...mlResult, hash, isValid });
    } catch (err) {
      console.error("Error:", err);
      setError(err.message || "Something went wrong.");
    }
  };

  return (
    <div className="ml-container">
      <h2 className="ml-title">Return Item Details</h2>

      <form className="ml-form" onSubmit={handleSubmit}>
        {/* Text Inputs */}
        <div className="ml-row">
          <label className="label-text">
            Receipt Text
            <input name="receipt_text" type="text" value={formData.receipt_text} onChange={handleChange} required />
          </label>
          <label className="label-text">
            Price ($)
            <input name="price" type="number" value={formData.price} onChange={handleChange} required />
          </label>
          <label className="label-text">
            Days Since Purchase
            <input name="days_since_purchase" type="number" value={formData.days_since_purchase} onChange={handleChange} required />
          </label>
        </div>

        <div className="ml-row">
          <label className="label-text">
            Return History
            <input name="return_history" type="number" value={formData.return_history} onChange={handleChange} required />
          </label>
          <label className="label-text">
            Returns Last 30 Days
            <input name="returns_last_30_days" type="number" value={formData.returns_last_30_days} onChange={handleChange} />
          </label>
          <label className="label-text">
            Account Age (days)
            <input name="account_age_days" type="number" value={formData.account_age_days} onChange={handleChange} required />
          </label>
        </div>

        {/* Dropdown Selects */}
        <div className="ml-row">
          <label className="label-text">
            Customer Tier
            <select name="customer_tier" value={formData.customer_tier} onChange={handleChange}>
              <option value="bronze">Bronze</option>
              <option value="silver">Silver</option>
              <option value="gold">Gold</option>
            </select>
          </label>

          <label className="label-text">
            Shipping Speed
            <select name="shipping_speed" value={formData.shipping_speed} onChange={handleChange}>
              <option value="standard">Standard</option>
              <option value="express">Express</option>
              <option value="overnight">Overnight</option>
            </select>
          </label>

          <label className="label-text">
            Payment Method
            <select name="payment_method" value={formData.payment_method} onChange={handleChange}>
              <option value="credit">Credit</option>
              <option value="debit">Debit</option>
              <option value="digital">Digital Wallet</option>
            </select>
          </label>
        </div>

        <div className="ml-row">
          <label className="label-text">
            Category
            <select name="category" value={formData.category} onChange={handleChange}>
              <option value="electronics">Electronics</option>
              <option value="clothing">Clothing</option>
              <option value="home">Home</option>
            </select>
          </label>

          <label className="label-text">
            Hour of Return
            <select name="hour_of_return" value={formData.hour_of_return} onChange={handleChange}>
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={i}>{`Hour: ${i}`}</option>
              ))}
            </select>
          </label>

          <div className="ml-checkbox-col">
            <label className="label-text checkbox-group">
              <input name="geo_mismatch" type="checkbox" checked={formData.geo_mismatch} onChange={handleChange} />
              Geo mismatch
            </label>
            <label className="label-text checkbox-group">
              <input name="is_weekend" type="checkbox" checked={formData.is_weekend} onChange={handleChange} />
              Weekend return
            </label>
          </div>
        </div>

        <button type="submit" className="ml-button">🧠 Predict Risk</button>
      </form>

      {result && (
        <div className="ml-result">
          <h3>✅ Prediction Result</h3>
          <p><strong>Receipt Hash:</strong> <code>{result.hash}</code></p>
          <p><strong>Risk Level:</strong> {result.risk_level}</p>
          <p><strong>Score:</strong> {result.risk_score}</p>
          <p><strong>Recommendation:</strong> {result.recommendation}</p>
        </div>
      )}

      {error && <p className="ml-error">{error}</p>}
    </div>
  );
};

export default MLForm;
