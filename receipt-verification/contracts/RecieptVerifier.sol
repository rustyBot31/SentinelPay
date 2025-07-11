// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReceiptVerifier {
    mapping(bytes32 => bool) public validReceipts;

    // Add a receipt hash (e.g., from Walmart system)
    function addReceiptHash(bytes32 receiptHash) public {
        validReceipts[receiptHash] = true;
    }

    // Verify if a given receipt hash exists
    function isValidReceipt(bytes32 receiptHash) public view returns (bool) {
        return validReceipts[receiptHash];
    }
}
