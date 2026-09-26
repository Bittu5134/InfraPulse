import React, { useState } from "react";
import { Send, Loader2, X, AlertCircle } from "lucide-react";
import ImageUploader from "./ImageUploader";

export default function ComplaintForm({ onSubmit, submitting, error, onCancel }) {
  const [form, setForm] = useState({
    name: "",
    address: "",
    description: ""
  });
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [localError, setLocalError] = useState("");

  const handlePhotoSelect = (file) => {
    setPhoto(file);
    setPhotoPreview(URL.createObjectURL(file));
    setLocalError("");
  };

  const handlePhotoClear = () => {
    setPhoto(null);
    if (photoPreview) {
      URL.revokeObjectURL(photoPreview);
      setPhotoPreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!photo) {
      setLocalError("Please upload a photo of the defect.");
      return;
    }
    setLocalError("");

    const fd = new FormData();
    fd.append("name", form.name);
    fd.append("address", form.address);
    fd.append("description", form.description);
    fd.append("photo", photo);

    const success = await onSubmit(fd);
    if (success) {
      setForm({ name: "", address: "", description: "" });
      handlePhotoClear();
    }
  };

  return (
    <div className="clean-form-card">
      <div className="form-card-top">
        <div>
          <h2 className="clean-form-title">Submit a Complaint</h2>
          <p className="clean-form-sub">
            Document an infrastructure issue. It will be automatically classified and added to the repair queue.
          </p>
        </div>
        {onCancel && (
          <button 
            type="button" 
            className="btn-close-form" 
            onClick={onCancel}
            title="Close form"
          >
            <X size={16} />
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="clean-form-stack">
        <div className="form-row-2">
          <div className="clean-input-group">
            <label className="clean-label" htmlFor="input-name">Your Name</label>
            <input
              id="input-name"
              type="text"
              className="clean-input"
              placeholder="e.g. John Smith"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            />
          </div>

          <div className="clean-input-group">
            <label className="clean-label" htmlFor="input-address">Defect Location</label>
            <input
              id="input-address"
              type="text"
              className="clean-input"
              placeholder="e.g. Building B, 2nd Floor Corridor"
              value={form.address}
              onChange={(e) => setForm({ ...form, address: e.target.value })}
              required
            />
          </div>
        </div>

        <div className="clean-input-group">
          <div className="label-with-hint">
            <label className="clean-label" htmlFor="input-desc">Description (Optional)</label>
            <span className="char-count">{form.description.length}/300</span>
          </div>
          <textarea
            id="input-desc"
            className="clean-textarea"
            rows={3}
            maxLength={300}
            placeholder="Briefly describe the defect (e.g. water pooling on floor, peeling paint on east wall)..."
            value={form.description}
            onChange={(e) => setForm({ ...form, description: e.target.value })}
          />
        </div>

        <div className="clean-input-group">
          <label className="clean-label">Defect Photograph</label>
          <ImageUploader
            photo={photo}
            photoPreview={photoPreview}
            onPhotoSelect={handlePhotoSelect}
            onPhotoClear={handlePhotoClear}
            error={localError}
          />
        </div>

        {(error || localError) && (
          <div className="clean-form-error">
            <AlertCircle size={14} />
            <span>{error || localError}</span>
          </div>
        )}

        <div className="form-action-row">
          {onCancel && (
            <button 
              type="button" 
              className="btn-cancel-action"
              onClick={onCancel}
            >
              Cancel
            </button>
          )}
          <button 
            type="submit" 
            className="btn-submit-action" 
            disabled={submitting}
          >
            {submitting ? (
              <>
                <Loader2 size={15} className="spinning-icon" />
                <span>Classifying & Submitting...</span>
              </>
            ) : (
              <>
                <span>Submit Complaint</span>
                <Send size={14} />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
