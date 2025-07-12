import React from 'react';
import { connectWallet } from '../utils/blockchain';

export default function ConnectWallet({ setWalletConnected }) {
  const handleConnect = async () => {
    const success = await connectWallet();
    setWalletConnected(success);
  };

  return (
    <button onClick={handleConnect}>
      Connect Wallet
    </button>
  );
}
