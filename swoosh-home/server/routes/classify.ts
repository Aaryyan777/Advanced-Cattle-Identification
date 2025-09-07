import { RequestHandler } from "express";
import type {
  ClassificationRequest,
  ClassificationResponse,
  ApiError,
} from "@shared/api";

async function forwardToInference(
  body: ClassificationRequest
): Promise<{ status: number; json: ClassificationResponse | ApiError }> {
  const url = process.env.INFERENCE_URL;
  if (!url) {
    // Simulate a response if no inference URL is provided
    console.warn(
      "INFERENCE_URL not set. Simulating a fake model response. Set INFERENCE_URL to your model's endpoint."
    );
    await new Promise((res) => setTimeout(res, 800)); // Simulate network delay
    const fakePredictions = [
      { label: "Fake Breed 1", confidence: 0.78 },
      { label: "Fake Breed 2", confidence: 0.12 },
      { label: "Fake Breed 3", confidence: 0.05 },
    ].slice(0, body.topK ?? 5);

    return {
      status: 200,
      json: {
        model: "simulated-model/v1",
        latencyMs: 800,
        predictions: fakePredictions,
      },
    };
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = process.env.INFERENCE_API_KEY;
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const resp = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const json = await resp.json();
  return { status: resp.status, json } as any;
}

export const handleClassify: RequestHandler = async (req, res) => {
  try {
    const body = req.body as ClassificationRequest | undefined;
    if (!body?.imageBase64 || typeof body.imageBase64 !== "string") {
      const err: ApiError = {
        code: "BAD_REQUEST",
        message: "imageBase64 is required",
      };
      return res.status(400).json(err);
    }

    const started = Date.now();
    const result = await forwardToInference({
      imageBase64: body.imageBase64,
      topK: body.topK ?? 5,
    });

    if (result.status >= 200 && result.status < 300) {
      const data = result.json as ClassificationResponse;
      // Ensure required fields
      const response: ClassificationResponse = {
        model: data.model ?? "external",
        latencyMs: data.latencyMs ?? Date.now() - started,
        predictions: (data.predictions ?? []).slice(0, body.topK ?? 5),
      };
      return res.status(200).json(response);
    }

    // Proxy error
    return res.status(result.status).json(result.json);
  } catch (e) {
    console.error("/api/classify error", e);
    const err: ApiError = {
      code: "INTERNAL_ERROR",
      message: "Unexpected server error",
    };
    return res.status(500).json(err);
  }
};