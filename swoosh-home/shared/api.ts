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

// Classification types
export interface ClassificationRequest {
  /** data URL string: e.g. "data:image/jpeg;base64,..." */
  imageBase64: string;
  /** Optional: number of top predictions to return */
  topK?: number;
}

export interface BreedPrediction {
  breed: string;
  confidence: number; // 0..1
}

export interface ClassificationResponse {
  model: string;
  latencyMs: number;
  predictions: BreedPrediction[]; // sorted desc by confidence
}

export interface ApiError {
  code: string;
  message: string;
}
