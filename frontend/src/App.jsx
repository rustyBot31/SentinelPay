import React, { useState } from 'react';
import ConnectWallet from './components/ConnectWallet';
import ReceiptInput from './components/ReceiptInput';
import HashResult from './components/HashResult';
import MLForm from './components/MLForm';
import './App.css';

function App() {
  const [walletConnected, setWalletConnected] = useState(false);
  const [result, setResult] = useState(null);

  return (
    <div className="app-container" style={{ minHeight: "100vh", backgroundColor: 'white' }}>
      <img
        src="https://img.favpng.com/16/25/2/logo-vector-graphics-brand-walmart-portable-network-graphics-png-favpng-RNmwyt5haE5GzqfvtZfVWaRk3.jpg"
        alt="Walmart Logo"
        className="walmart-logo"
      />
      <h1 className="title">SentinelPay Receipt Verifier</h1>

      {!walletConnected ? (
        <ConnectWallet setWalletConnected={setWalletConnected} />
      ) : (
        <div className="content-wrapper">
          <div className="yellow-box">
            <MLForm />
          </div>
          <div className="yellow-box">
            <ReceiptInput setResult={setResult} />
            <HashResult result={result} />
          </div>
        </div>
      )}
    </div>
  );
}
/*<div className="yellow-box">
    <ReceiptInput setResult={setResult} />
    <HashResult result={result} />
  </div>*/
export default App;
