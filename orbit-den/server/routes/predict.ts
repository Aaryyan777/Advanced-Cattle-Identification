import type { RequestHandler } from "express";
import type { PredictRequest, PredictResponse, BreedPrediction } from "@shared/api";

export const handlePredict: RequestHandler = async (req, res) => {
  try {
    const body = req.body as PredictRequest | undefined;
    if (!body?.imageBase64) {
      return res.status(400).json({ error: "imageBase64 is required" });
    }

    const endpoint = process.env.MODEL_ENDPOINT;
    if (!endpoint) {
      return res.status(501).json({ error: "MODEL_ENDPOINT not configured" });
    }

    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageBase64: body.imageBase64 }),
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      return res.status(502).json({ error: `Upstream error: ${upstream.status}`, detail: text?.slice(0, 500) });
    }

    const data = (await upstream.json()) as PredictResponse;
    const predictions: BreedPrediction[] = Array.isArray(data?.predictions)
      ? data.predictions.map((p: any) => ({ breed: String(p.label), confidence: Number(p.confidence) }))
      : [];

    return res.status(200).json({ predictions, meta: data?.meta } satisfies PredictResponse);
  } catch (e) {
    return res.status(500).json({ error: "Unexpected server error" });
  }
};
