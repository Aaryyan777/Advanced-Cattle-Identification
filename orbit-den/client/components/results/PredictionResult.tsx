import type { BreedPrediction } from "@shared/api";
import { BREEDS, normalizeBreedName } from "@/data/breeds";

export default function PredictionResult({ predictions }: { predictions: BreedPrediction[] }) {
  if (!predictions?.length) {
    return (
      <div className="text-center border-4 border-dashed border-destructive p-8">
        <div className="text-2xl font-extrabold uppercase">No Cattle Detected</div>
        <p className="mt-2 text-muted-foreground">
          The image was successfully analyzed and does not appear to contain a recognized cattle breed. Please try a different photo.
        </p>
      </div>
    );
  }
  const top = predictions[0];
  const key = normalizeBreedName(top.breed);
  const info = BREEDS[key];

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div>
        <h2 className="text-3xl font-extrabold uppercase leading-none">
          Detected: <span className="bg-primary px-2 border-4 border-foreground">{top.breed}</span>
        </h2>
        <div className="mt-2 text-sm">Confidence: {(top.confidence * 100).toFixed(1)}%</div>
        {info && (
          <div className="mt-4 space-y-2">
            <div className="text-lg font-bold">About {info.name}</div>
            <p className="leading-relaxed">{info.description}</p>
            <div className="text-sm">Origin: {info.origin} • Purpose: {info.purpose.join(", ")}</div>
          </div>
        )}

        {predictions.length > 1 && (
          <div className="mt-6">
            <div className="text-sm uppercase font-bold">Other likely breeds</div>
            <ul className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2">
              {predictions.slice(1).map((p) => (
                <li key={p.breed} className="border-4 border-foreground px-2 py-1 bg-secondary">
                  <div className="flex items-center justify-between gap-2">
                    <span>{p.breed}</span>
                    <span className="text-xs">{(p.confidence * 100).toFixed(0)}%</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div>
        <div className="text-sm uppercase font-bold">Reference images</div>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {(info?.images ?? []).slice(0, 6).map((img, idx) => (
            <div key={idx} className="border-4 border-foreground bg-muted overflow-hidden">
              <img src={img.url} alt={`${info?.name} ${idx + 1}`} className="w-full h-32 object-cover" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
