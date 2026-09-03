import { defineRailway, project, service } from "railway/iac";

// Last resort for a per-service CaC repo. Prefer one .railway file for the
// project and drop this if you later combine services into that file.
export const partial = "InfraPulse";

export default defineRailway(() => {
  const InfraPulse = service("InfraPulse", {
    start: "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1",
    healthcheck: "/health",
    healthcheckTimeout: 100,
    // dockerfilePath from CaC: "Dockerfile"
    // builder from CaC: "DOCKERFILE"
  });
  return project("InfraPulse", {
    resources: [InfraPulse],
  });
});
