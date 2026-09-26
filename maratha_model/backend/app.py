"""
HF Spaces entry point (Gradio SDK).
Mounts the InfraPulse FastAPI app inside Gradio's server so all
existing API routes (/auth, /complaints, /staff, /ws/queue, etc.)
are fully accessible. Gradio just provides the runner HF expects.
"""
import uvicorn
import gradio as gr

from app.main import app as fastapi_app  # our existing FastAPI app

# Minimal Gradio UI — just a status page at /ui
with gr.Blocks(title="InfraPulse API", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏗️ InfraPulse API
        **Status:** Running ✅

        This Space hosts the InfraPulse backend REST API.
        All endpoints are accessible directly at the root URL.

        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | `/health` | GET | Health check |
        | `/auth/login` | POST | Login |
        | `/complaints` | GET/POST | Complaints |
        | `/staff` | GET | Staff list |
        | `/ws/queue` | WS | Real-time queue |
        """
    )

# Mount Gradio status UI at /ui — FastAPI handles everything else
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
