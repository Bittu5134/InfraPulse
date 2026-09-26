---
title: InfraPulse API
emoji: 🏗️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# InfraPulse Backend API

FastAPI backend for the InfraPulse infrastructure complaint management system.

## Endpoints

- `GET /health` — health check
- `POST /auth/login` — staff/user login
- `GET /complaints` — list complaints
- `POST /complaints` — submit a complaint (with image)
- `WS /ws/queue` — real-time queue updates
