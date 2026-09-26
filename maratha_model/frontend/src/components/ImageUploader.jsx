import React, { useState, useRef } from "react";
import { Camera, X, Image as ImageIcon } from "lucide-react";

export default function ImageUploader({ photo, photoPreview, onPhotoSelect, onPhotoClear, error }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        onPhotoSelect(file);
      }
    }
  };

  return (
    <div className="clean-uploader">
      {!photoPreview ? (
        <div
          className={`clean-dropzone ${isDragging ? "is-dragging" : ""} ${error ? "has-error" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && onPhotoSelect(e.target.files[0])}
            accept="image/*"
            className="hidden-file-input"
          />
          <div className="uploader-icon-circle">
            <Camera size={20} />
          </div>
          <span className="dropzone-text">
            <strong>Click to upload</strong> or drag a photo here
          </span>
          <span className="dropzone-sub">Supports JPEG, PNG, or WebP</span>
        </div>
      ) : (
        <div className="clean-preview-box">
          <img src={photoPreview} alt="Defect preview" className="clean-preview-img" />
          <div className="preview-info-col">
            <span className="preview-filename">{photo?.name || "defect_photo.jpg"}</span>
            <span className="preview-filesize">
              {photo?.size ? `${(photo.size / (1024 * 1024)).toFixed(2)} MB` : "Photo attached"}
            </span>
          </div>
          <button 
            type="button" 
            className="btn-clear-photo"
            onClick={onPhotoClear}
            title="Remove photo"
          >
            <X size={15} />
          </button>
        </div>
      )}
      {error && <span className="clean-error-text">{error}</span>}
    </div>
  );
}
