import React, { useState } from 'react';
import { storeReceiptHash, verifyReceiptHash } from '../utils/blockchain';

export default function ReceiptInput({ setResult }) {
  const [receiptText, setReceiptText] = useState("");

  const handleStore = async () => {
    const hash = await storeReceiptHash(receiptText);
    setResult({ hash, isValid: true });
  };

  const handleVerify = async () => {
    const res = await verifyReceiptHash(receiptText);
    setResult(res);
  };

  return (
    <div>
      <input
        type="text"
        placeholder="Enter receipt text"
        value={receiptText}
        onChange={(e) => setReceiptText(e.target.value)}
      />
      <button onClick={handleStore}>Store Receipt</button>
      <button onClick={handleVerify}>Verify Receipt</button>
    </div>
  );
}
