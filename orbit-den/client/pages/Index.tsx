import { useMemo, useState } from "react";
import ImageUploader from "@/components/capture/ImageUploader";
import CameraCapture from "@/components/capture/CameraCapture";
import PredictionResult from "@/components/results/PredictionResult";
import { analyzeHeuristic, fileToDataUrl } from "@/lib/image";
import { predictImage } from "@/lib/api";
import type { BreedPrediction } from "@shared/api";

export default function Index() {
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [predictions, setPredictions] = useState<BreedPrediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const onAnalyze = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setPredictions(null);
    try {
      const base64 = await fileToDataUrl(file);
      try {
        const res = await predictImage(base64);
        setPredictions(res.predictions);
      } catch {
        const heur = await analyzeHeuristic(file);
        setPredictions(heur);
      }
    } catch (e) {
      setError("Failed to analyze image.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-10">
      <section className="grid md:grid-cols-[1.1fr,1fr] gap-8 items-start">
        <div className="space-y-4">
          <h1 className="text-5xl md:text-6xl font-extrabold uppercase leading-[0.95]">
            Brutalist Cattle Breed Classifier
          </h1>
          <p className="max-w-prose text-lg">
            Upload or capture a photo. Our model identifies and classifies cattle breeds, then shows rich information and reference images.
          </p>
          <div className="flex gap-2">
            <button
              className={`px-4 py-2 border-4 border-foreground uppercase ${mode === "upload" ? "bg-primary" : "bg-secondary"}`}
              onClick={() => setMode("upload")}
            >
              Upload
            </button>
            <button
              className={`px-4 py-2 border-4 border-foreground uppercase ${mode === "camera" ? "bg-primary" : "bg-secondary"}`}
              onClick={() => setMode("camera")}
            >
              Camera
            </button>
          </div>

          {mode === "upload" ? (
            <ImageUploader onSelect={setFile} disabled={busy} />
          ) : (
            <CameraCapture onCapture={setFile} disabled={busy} />
          )}

          {file && (
            <div className="flex items-center gap-3">
              <button
                className="bg-primary text-primary-foreground px-6 py-3 border-4 border-foreground uppercase tracking-wide"
                onClick={onAnalyze}
                disabled={busy}
              >
                {busy ? "Analyzing…" : "Analyze"}
              </button>
              <button
                className="bg-secondary px-4 py-3 border-4 border-foreground uppercase"
                onClick={() => { setFile(null); setPredictions(null); setError(null); }}
                disabled={busy}
              >
                Reset
              </button>
            </div>
          )}

          {error && <div className="text-destructive font-bold">{error}</div>}
        </div>

        <div className="border-4 border-foreground bg-muted min-h-[260px] flex items-center justify-center overflow-hidden">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
          ) : (
            <div className="text-center text-sm uppercase p-8">Your image preview appears here</div>
          )}
        </div>
      </section>

      {predictions && (
        <section>
          <PredictionResult predictions={predictions} />
        </section>
      )}

      <section id="how" className="border-t-4 border-foreground pt-8">
        <h3 className="text-xl font-bold uppercase">How it works</h3>
        <ol className="list-decimal pl-5 mt-2 space-y-1">
          <li>Upload or take a photo of a single cow, side-on if possible.</li>
          <li>We run your image through a trained model (or a local heuristic if no model is connected).</li>
          <li>We display the most likely breeds with information and images.</li>
        </ol>
      </section>

      <section id="about" className="border-t-4 border-foreground pt-8">
        <h3 className="text-xl font-bold uppercase">Connect your model</h3>
        <p className="mt-2 max-w-prose">
          Provide your model endpoint via the MODEL_ENDPOINT environment variable on the server. The app will send base64-encoded images and
          display the returned predictions.
        </p>
      </section>
    </div>
  );
}
