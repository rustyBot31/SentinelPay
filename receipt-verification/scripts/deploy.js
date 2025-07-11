async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying contracts with the account:", deployer.address);

  const ReceiptVerifier = await ethers.getContractFactory("ReceiptVerifier");
  const receiptVerifier = await ReceiptVerifier.deploy();

  await receiptVerifier.waitForDeployment();

  console.log("ReceiptVerifier deployed to:", await receiptVerifier.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
