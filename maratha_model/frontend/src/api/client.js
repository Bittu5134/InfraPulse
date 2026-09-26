const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function authHeaders() {
  const token = localStorage.getItem("infrapulse_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function signup(role, data) {
  const res = await fetch(`${API_URL}/auth/signup/${role}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handle(res);
}

export async function login(data) {
  const res = await fetch(`${API_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handle(res);
}

export async function submitComplaint(formData) {
  const res = await fetch(`${API_URL}/complaints`, {
    method: "POST",
    headers: { ...authHeaders() },
    body: formData,
  });
  return handle(res);
}

export async function myComplaints() {
  const res = await fetch(`${API_URL}/complaints/mine`, {
    headers: { ...authHeaders() },
  });
  return handle(res);
}

export async function staffQueue() {
  const res = await fetch(`${API_URL}/staff/queue`, {
    headers: { ...authHeaders() },
  });
  return handle(res);
}

export async function updateStatus(complaintId, status) {
  const res = await fetch(`${API_URL}/staff/complaints/${complaintId}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ status }),
  });
  return handle(res);
}

export function imageUrl(path) {
  return `${API_URL}${path}`;
}

export const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";
