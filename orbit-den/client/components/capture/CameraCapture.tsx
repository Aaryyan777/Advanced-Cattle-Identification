import { useEffect, useRef, useState } from "react";

export type CameraCaptureProps = {
  onCapture: (file: File) => void;
  disabled?: boolean;
};

export default function CameraCapture({ onCapture, disabled }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        if (videoRef.current && streamRef.current) {
          videoRef.current.srcObject = streamRef.current;
          await videoRef.current.play();
          setReady(true);
        }
      } catch (e) {
        setError("Camera unavailable. Check permissions.");
      }
    })();
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const capture = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    const file = await dataUrlToFile(dataUrl, "capture.jpg");
    onCapture(file);
  };

  return (
    <div className="border-4 border-foreground bg-secondary p-2 shadow-brutal">
      <div className="relative aspect-video bg-muted">
        {error ? (
          <div className="flex items-center justify-center h-full text-sm">{error}</div>
        ) : (
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
        )}
      </div>
      <div className="mt-2 flex gap-2">
        <button
          className="w-full bg-primary text-primary-foreground px-4 py-2 border-4 border-foreground uppercase tracking-wide"
          onClick={capture}
          disabled={!ready || disabled}
        >
          Capture
        </button>
      </div>
    </div>
  );
}

async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const buf = await res.arrayBuffer();
  return new File([buf], filename, { type: "image/jpeg" });
}
