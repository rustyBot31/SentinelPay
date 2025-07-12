import React, { useState } from "react";
import { verifyReceiptHash } from "../utils/blockchain";
import { predictReturnRisk } from "../services/mlApi";

const MLForm = () => {
  const [formData, setFormData] = useState({
    receipt_text: "", // NEW field
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
      const receiptText = formData.receipt_text;
      const { hash, isValid } = await verifyReceiptHash(receiptText);
      console.log("🔍 Verifying hash:", hash, "isValid:", isValid);

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
    <div>
      <h2>Return Risk Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input
          name="receipt_text"
          type="text"
          placeholder="Receipt string (must match stored)"
          value={formData.receipt_text}
          onChange={handleChange}
          required
        />
        <input name="price" type="number" placeholder="Price" value={formData.price} onChange={handleChange} required />
        <input name="days_since_purchase" type="number" placeholder="Days Since Purchase" value={formData.days_since_purchase} onChange={handleChange} required />
        <input name="return_history" type="number" placeholder="Return History" value={formData.return_history} onChange={handleChange} required />
        <input name="returns_last_30_days" type="number" placeholder="Returns Last 30 Days" value={formData.returns_last_30_days} onChange={handleChange} />
        <input name="account_age_days" type="number" placeholder="Account Age (Days)" value={formData.account_age_days} onChange={handleChange} required />

        <label>
          <input name="geo_mismatch" type="checkbox" checked={formData.geo_mismatch} onChange={handleChange} />
          Geo mismatch?
        </label>

        <label>
          <input name="is_weekend" type="checkbox" checked={formData.is_weekend} onChange={handleChange} />
          Weekend return?
        </label>

        <select name="hour_of_return" value={formData.hour_of_return} onChange={handleChange}>
          {Array.from({ length: 24 }, (_, i) => (
            <option key={i} value={i}>{`Hour: ${i}`}</option>
          ))}
        </select>

        <select name="customer_tier" value={formData.customer_tier} onChange={handleChange}>
          <option value="bronze">Bronze</option>
          <option value="silver">Silver</option>
          <option value="gold">Gold</option>
        </select>

        <select name="shipping_speed" value={formData.shipping_speed} onChange={handleChange}>
          <option value="standard">Standard</option>
          <option value="express">Express</option>
          <option value="overnight">Overnight</option>
        </select>

        <select name="payment_method" value={formData.payment_method} onChange={handleChange}>
          <option value="credit">Credit</option>
          <option value="debit">Debit</option>
          <option value="digital">Digital Wallet</option>
        </select>

        <select name="category" value={formData.category} onChange={handleChange}>
          <option value="electronics">Electronics</option>
          <option value="clothing">Clothing</option>
          <option value="home">Home</option>
        </select>

        <button type="submit">Predict Risk</button>
      </form>

      {result && (
        <div style={{ marginTop: 20 }}>
          <h3>✅ Prediction Result</h3>
          <p>Receipt Hash: <code>{result.hash}</code></p>
          <p>Risk Level: <strong>{result.risk_level}</strong></p>
          <p>Score: <strong>{result.risk_score}</strong></p>
          <p>Recommendation: <strong>{result.recommendation}</strong></p>
        </div>
      )}

      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default MLForm;
