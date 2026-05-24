import React, { useRef, useState } from "react";

interface Props {
  onFile: (b64: string, name: string) => void;
}

const FileUpload: React.FC<Props> = ({ onFile }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is like "data:application/pdf;base64,...." – strip prefix
      const b64 = result.split(",")[1];
      onFile(b64, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const openFileDialog = () => {
    inputRef.current?.click();
  };

  return (
    <div
      className={`flex items-center justify-center border-2 border-dashed p-2 cursor-pointer ${
        dragActive ? "border-blue-500" : "border-gray-300"
      }`}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      onClick={openFileDialog}
    >
      <input
        type="file"
        ref={inputRef}
        style={{ display: "none" }}
        accept="image/*,application/pdf"
        onChange={handleChange}
      />
      <span className="text-sm text-gray-600">Upload image or PDF</span>
    </div>
  );
};

export default FileUpload;
