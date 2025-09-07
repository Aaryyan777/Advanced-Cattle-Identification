import { useCallback, useRef, useState } from "react";

export type ImageUploaderProps = {
  onSelect: (file: File) => void;
  disabled?: boolean;
};

export default function ImageUploader({ onSelect, disabled }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) onSelect(file);
  }, [onSelect, disabled]);

  const onChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) onSelect(file);
  }, [onSelect]);

  return (
    <div
      className={`border-4 border-foreground bg-secondary p-4 text-center cursor-pointer select-none ${dragOver ? "bg-accent" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="text-xl font-bold uppercase">Drop image here</div>
      <div className="mt-1 text-sm">or click to browse</div>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onChange} disabled={disabled} />
    </div>
  );
}
