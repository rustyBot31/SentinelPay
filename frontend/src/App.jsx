import React, { useState } from 'react';
import ConnectWallet from './components/ConnectWallet';
import ReceiptInput from './components/ReceiptInput';
import HashResult from './components/HashResult';

function App() {
  const [walletConnected, setWalletConnected] = useState(false);
  const [result, setResult] = useState(null);

  return (
    <div style={{ padding: 20 }}>
      <h1>🧾 SentinelPay Receipt Verifier</h1>
      {!walletConnected ? (
        <ConnectWallet setWalletConnected={setWalletConnected} />
      ) : (
        <>
          <ReceiptInput setResult={setResult} />
          <HashResult result={result} />
        </>
      )}
    </div>
  );
}

export default App;
