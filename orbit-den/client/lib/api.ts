import type { PredictResponse, PredictRequest } from "@shared/api";

export async function predictImage(imageBase64: string): Promise<PredictResponse> {
  const payload: PredictRequest = { imageBase64 };
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Prediction failed: ${res.status}`);
  }
  return (await res.json()) as PredictResponse;
}
