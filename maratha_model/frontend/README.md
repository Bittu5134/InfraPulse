# InfraPulse frontend

React (Vite) app with separate User and Staff portals.

## Setup

```bash
npm install
cp .env.example .env   # point VITE_API_URL / VITE_WS_URL at the backend
npm run dev
```

## Routes

- `/` — user login/signup
- `/staff/login` — staff login/signup (with category selection)
- `/user/dashboard` — submit + track complaints (protected)
- `/staff/dashboard` — live priority queue for the staff member's category (protected)
