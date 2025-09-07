export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function dataUrlToImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

export async function analyzeHeuristic(
  file: File,
): Promise<{ breed: string; confidence: number }[]> {
  const dataUrl = await fileToDataUrl(file);
  const img = await dataUrlToImage(dataUrl);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  const w = 160;
  const h = Math.max(1, Math.round((img.height / img.width) * w));
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  let sumL = 0,
    sumL2 = 0,
    sumR = 0,
    sumG = 0,
    sumB = 0;
  const n = w * h;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b; // relative luminance
    sumL += l;
    sumL2 += l * l;
    sumR += r;
    sumG += g;
    sumB += b;
  }
  const avgL = sumL / n;
  const stdL = Math.sqrt(Math.max(0, sumL2 / n - avgL * avgL));
  const avgR = sumR / n;
  const avgG = sumG / n;
  const avgB = sumB / n;

  // Simple heuristics: choose top matches
  const scores: Record<string, number> = {
    Holstein: 0,
    Angus: 0,
    Hereford: 0,
    Charolais: 0,
    "Brown Swiss": 0,
    Jersey: 0,
    Brahman: 0,
    Simmental: 0,
  };

  // Dark overall -> Angus
  if (avgL < 0.35) scores["Angus"] += (0.35 - avgL) * 2;
  // Very light overall -> Charolais (white/cream)
  if (avgL > 0.75) scores["Charolais"] += (avgL - 0.75) * 2;
  // Red-ish dominance -> Hereford / Simmental
  if (avgR > avgG && avgR > avgB) {
    scores["Hereford"] += (avgR - Math.max(avgG, avgB)) * 1.2;
    scores["Simmental"] += (avgR - Math.max(avgG, avgB)) * 0.6;
  }
  // High contrast (spots) -> Holstein
  scores["Holstein"] += Math.min(1, stdL * 2);
  // Warm mid tones -> Jersey / Brown Swiss
  const warm = avgR > avgB && avgL > 0.45 && avgL < 0.7;
  if (warm) {
    scores["Jersey"] += 0.8;
    scores["Brown Swiss"] += 0.5;
  }
  // Slightly grey and mid-light -> Brahman
  const greyish = Math.abs(avgR - avgG) < 0.05 && Math.abs(avgG - avgB) < 0.05;
  if (greyish && avgL > 0.5 && avgL < 0.75) scores["Brahman"] += 0.6;

  const entries = Object.entries(scores)
    .map(([breed, score]) => ({ breed, confidence: Math.max(0, Math.min(1, score)) }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  // Normalize confidences to sum to 1 for display
  const total = entries.reduce((sum, e) => sum + e.confidence, 0) || 1;
  return entries.map((e) => ({ breed: e.breed, confidence: e.confidence / total }));
}
