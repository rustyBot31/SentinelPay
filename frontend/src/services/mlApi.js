export async function predictReturnRisk(data) {
  const response = await fetch("http://localhost:5000/predict-return-risk", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Something went wrong.");
  }

  return await response.json();
}
