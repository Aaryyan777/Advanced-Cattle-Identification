import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { BreedPrediction, ClassificationResponse, ApiError } from "@shared/api";
import { Camera, Upload, RefreshCw, AlertTriangle, Info, X, Aperture } from "lucide-react";

type Status = "idle" | "classifying" | "success" | "error" | "not_configured";

export default function Index() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<BreedPrediction[] | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const reset = () => {
    setImageBase64(null);
    setPredictions(null);
    setStatus("idle");
    setError(null);
    setIsCameraOpen(false); // This will trigger the useEffect cleanup
  };

  // This useEffect handles the camera stream setup and cleanup
  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera: ", err);
        setError("Could not access the camera. Please ensure you have given the necessary permissions.");
        setStatus("error");
        setIsCameraOpen(false);
      }
    };

    if (isCameraOpen) {
      startCamera();
    }

    // Cleanup function: this runs when the component unmounts or isCameraOpen becomes false
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isCameraOpen]); // This effect runs whenever isCameraOpen changes

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const context = canvas.getContext("2d");
      if (context) {
        context.drawImage(videoRef.current, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg");
        setImageBase64(dataUrl);
        classify(dataUrl);
        setIsCameraOpen(false); // Close camera after capture
      } else {
        setError("Failed to capture image from camera feed.");
        setStatus("error");
        setIsCameraOpen(false);
      }
    }
  };

  const readFile = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      setStatus("error");
      return;
    }
    const dataUrl = await readFile(file);
    setImageBase64(dataUrl);
    await classify(dataUrl);
  };

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    await handleFiles(e.dataTransfer.files);
  };

  const onPaste = useCallback(async (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) await handleFiles({ 0: file, length: 1, item: () => file } as any);
        break;
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener("paste", onPaste as any);
    return () => {
      window.removeEventListener("paste", onPaste as any);
    };
  }, [onPaste]);

  const classify = async (dataUrl: string) => {
    try {
      setStatus("classifying");
      setError(null);
      const resp = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64: dataUrl, topK: 5 }),
      });

      if (!resp.ok) {
        const err = (await resp.json()) as ApiError;
        if (resp.status === 503 && err?.code === "MODEL_NOT_CONFIGURED") {
          setStatus("not_configured");
          setError(
            "Model backend not configured. Set INFERENCE_URL (and optionally INFERENCE_API_KEY) for the server.",
          );
          return;
        }
        setStatus("error");
        setError(err?.message || "Failed to classify the image.");
        return;
      }

      const data = (await resp.json()) as ClassificationResponse;
      setPredictions(data.predictions);
      setStatus("success");
    } catch (e) {
      setStatus("error");
      setError("Network error while classifying the image.");
    }
  };

  if (isCameraOpen) {
    return (
      <div className="fixed inset-0 bg-black/80 z-50 flex flex-col items-center justify-center p-4">
        <video ref={videoRef} autoPlay playsInline className="w-full h-full max-h-[80vh] max-w-full object-contain"></video>
        <div className="mt-4 flex items-center gap-4">
            <Button onClick={captureImage} className="border-4 border-foreground bg-primary text-primary-foreground brutal-shadow hover:translate-x-[1px] hover:translate-y-[1px]">
                <Aperture className="mr-2" /> Capture
            </Button>
            <Button onClick={() => setIsCameraOpen(false)} variant="secondary" className="border-4 border-foreground brutal-shadow hover:translate-x-[1px] hover:translate-y-[1px]">
                <X className="mr-2" /> Close Camera
            </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-grid" aria-hidden="true" />
      <section className="container py-10 md:py-16">
        <div className="mb-8 md:mb-12 text-center">
          <div className="inline-block border-4 border-foreground px-4 py-2 brutal-shadow">
            <span className="text-xs tracking-widest uppercase">Cattle Breed Identification</span>
          </div>
          <h1 className="mt-6 text-4xl md:text-6xl font-extrabold tracking-tight leading-none">
            CATTLE.VISION
          </h1>
          <p className="mt-4 md:mt-6 max-w-2xl mx-auto text-base md:text-lg text-muted-foreground">
            Drop a photo, paste from clipboard, or use your camera. Our model identifies cattle breed with confidence scores.
          </p>
        </div>

        <div className="grid gap-8 md:gap-10 lg:grid-cols-2">
          {/* Uploader */}
          <div
            id="uploader"
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            className="relative border-4 border-foreground bg-card p-4 md:p-6 brutal-shadow"
          >
            <div className="flex flex-col sm:flex-row gap-4 items-stretch">
              <div className="flex-1 min-h-[260px] border-4 border-foreground bg-secondary/30 grid place-items-center text-center p-4">
                {!imageBase64 ? (
                  <div className="space-y-3">
                    <div className="mx-auto grid place-items-center size-16 rounded-full border-4 border-dashed border-foreground">
                      <Upload className="size-7" />
                    </div>
                    <p className="font-medium">Drag & drop an image here</p>
                    <p className="text-sm text-muted-foreground">or paste directly from your clipboard</p>
                  </div>
                ) : (
                  <img
                    src={imageBase64}
                    alt="Selected cattle"
                    className="max-h-[380px] w-auto object-contain border-4 border-foreground"
                  />
                )}
              </div>

              <div className="sm:w-[220px] flex sm:flex-col gap-3">
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="border-4 border-foreground bg-primary text-primary-foreground brutal-shadow hover:translate-x-[1px] hover:translate-y-[1px]"
                >
                  <Upload className="mr-2" /> Upload
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => setIsCameraOpen(true)} // Simply set the state to true
                  className="border-4 border-foreground brutal-shadow hover:translate-x-[1px] hover:translate-y-[1px]"
                >
                  <Camera className="mr-2" /> Use camera
                </Button>
                <Button
                  variant="ghost"
                  onClick={reset}
                  disabled={status === "classifying" && !imageBase64}
                  className="border-4 border-foreground brutal-shadow hover:translate-x-[1px] hover:translate-y-[1px]"
                >
                  <RefreshCw className="mr-2" /> Reset
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFiles(e.target.files)}
                />
              </div>
            </div>

            <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
              <Info className="size-4" /> Paste with Ctrl/⌘ + V
            </div>
          </div>

          {/* Results */}
          <div className="border-4 border-foreground bg-card p-4 md:p-6 brutal-shadow min-h-[260px]">
            {status === "idle" && (
              <div className="h-full grid place-items-center text-center text-muted-foreground">
                <p>Upload a cattle photo to see breed predictions.</p>
              </div>
            )}

            {status === "classifying" && (
              <div className="space-y-4">
                <p className="font-semibold">Analyzing image…</p>
                <Progress value={66} />
              </div>
            )}

            {status === "error" && (
              <div className="flex items-start gap-3 border-4 border-destructive p-4 bg-destructive/10">
                <AlertTriangle className="mt-0.5" />
                <div>
                  <p className="font-semibold">Error</p>
                  <p className="text-sm text-muted-foreground">{error}</p>
                </div>
              </div>
            )}

            {status === "not_configured" && (
              <div className="flex items-start gap-3 border-4 border-yellow-500 p-4 bg-yellow-100">
                <AlertTriangle className="mt-0.5" />
                <div>
                  <p className="font-semibold">Model backend not configured</p>
                  <p className="text-sm text-muted-foreground">
                    The UI is ready. Configure INFERENCE_URL on the server to connect your Keras model inference endpoint.
                  </p>
                </div>
              </div>
            )}

            {status === "success" && predictions && (
              <>
                {predictions.length === 0 ? (
                  <div className="h-full grid place-items-center text-center">
                    <div className="flex items-start gap-3 border-4 border-destructive p-4 bg-destructive/10">
                      <AlertTriangle className="mt-0.5" />
                      <div>
                        <p className="font-semibold">No Cattle Detected</p>
                        <p className="text-sm text-muted-foreground">The image was successfully analyzed and does not appear to contain a recognized cattle breed. Please try a different photo.</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-5">
                    <div>
                      <p className="text-xs tracking-widest uppercase text-muted-foreground">Top prediction</p>
                      <h2 className="text-3xl md:text-4xl font-extrabold leading-none mt-1">
                        {predictions[0]?.label}
                      </h2>
                    </div>

                    <div className="space-y-4">
                      {predictions.map((p) => (
                        <div key={p.label} className="">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono text-sm">{p.label}</span>
                            <span className="font-mono text-sm">{Math.round(p.confidence * 100)}%</span>
                          </div>
                          <Progress value={Math.max(1, Math.round(p.confidence * 100))} className="h-3 border-4 border-foreground" />
                        </div>
                      ))}
                    </div>

                    <div className="pt-2 text-xs text-muted-foreground">
                      Predictions are model outputs and may be imperfect. Use professional judgment for critical decisions.
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
