import { ethers } from "ethers";
import ReceiptVerifier from "../artifacts/ReceiptVerifier.json"; // we'll add this next

const CONTRACT_ADDRESS = import.meta.env.VITE_CONTRACT_ADDRESS;

let provider;
let signer;
let contract;

export async function connectWallet() {
  if (window.ethereum) {
    provider = new ethers.BrowserProvider(window.ethereum);
    await window.ethereum.request({ method: "eth_requestAccounts" });
    signer = await provider.getSigner();
    contract = new ethers.Contract(CONTRACT_ADDRESS, ReceiptVerifier.abi, signer);
    return true;
  } else {
    alert("Please install MetaMask.");
    return false;
  }
}

export async function storeReceiptHash(receiptText) {
  const hash = ethers.keccak256(ethers.toUtf8Bytes(receiptText));
  const tx = await contract.addReceiptHash(hash);
  await tx.wait();
  return hash;
}

export async function verifyReceiptHash(receiptText) {
  const hash = ethers.keccak256(ethers.toUtf8Bytes(receiptText));
  const isValid = await contract.isValidReceipt(hash);
  return { hash, isValid };
}
