import React from 'react';

export default function HashResult({ result }) {
  if (!result) return null;

  return (
    <div>
      <p><strong>Hash:</strong> {result.hash}</p>
      <p style={{ color: result.isValid ? 'green' : 'red' }}>
        {result.isValid ? '✅ Receipt is valid' : '❌ Receipt is invalid'}
      </p>
    </div>
  );
}
