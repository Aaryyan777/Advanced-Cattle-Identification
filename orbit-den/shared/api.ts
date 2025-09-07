/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

export interface BreedPrediction {
  breed: string;
  confidence: number; // 0..1
}

export interface PredictRequest {
  // base64 encoded image data URL (data:image/jpeg;base64,....) or raw base64
  imageBase64: string;
}

export interface PredictResponse {
  // sorted descending by confidence
  predictions: BreedPrediction[];
  // optional additional metadata from model
  meta?: Record<string, unknown>;
}
