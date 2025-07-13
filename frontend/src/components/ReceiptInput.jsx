import React, { useState } from 'react';
import { storeReceiptHash, verifyReceiptHash } from '../utils/blockchain';
import './ReceiptInput.css'; // Make sure to create this CSS file

export default function ReceiptInput({ setResult }) {
  const [receiptText, setReceiptText] = useState("");

  const handleStore = async () => {
    if (!receiptText.trim()) return;
    const hash = await storeReceiptHash(receiptText);
    setResult({ hash, isValid: true });
  };

  const handleVerify = async () => {
    if (!receiptText.trim()) return;
    const res = await verifyReceiptHash(receiptText);
    setResult(res);
  };

  return (
    <div className="receipt-container">
      <h2 className="receipt-header">Blockchain Receipt Hashing</h2>
      <input
        type="text"
        className="receipt-input"
        placeholder="Enter receipt text"
        value={receiptText}
        onChange={(e) => setReceiptText(e.target.value)}
      />
      <div className="receipt-buttons">
        <button className="receipt-button" onClick={handleStore}>Store Receipt</button>
        <button className="receipt-button" onClick={handleVerify}>Verify Receipt</button>
      </div>
    </div>
  );
}
