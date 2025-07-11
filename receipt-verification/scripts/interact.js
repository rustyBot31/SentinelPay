require("dotenv").config();
const { ethers } = require("hardhat");
const { keccak256, toUtf8Bytes } = require("ethers"); // ✅ use this directly

async function main() {
  const contractAddress = process.env.CONTRACT_ADDRESS;

  // Connect to the deployed contract
  const ReceiptVerifier = await ethers.getContractFactory("ReceiptVerifier");
  const receiptVerifier = await ReceiptVerifier.attach(contractAddress);

  const receipt = "WALMART-ORDER-#123456";
  const hash = keccak256(toUtf8Bytes(receipt)); // ✅ fixed this line

  console.log("Hashing and storing receipt:", receipt);
  console.log("Hash:", hash);

  await receiptVerifier.addReceiptHash(hash);
  console.log("Receipt stored on blockchain ✅");

  const isValid = await receiptVerifier.isValidReceipt(hash);
  console.log("Receipt valid?", isValid);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
