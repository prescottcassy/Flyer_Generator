const API_BASE = import.meta.env.VITE_API_URL;

export async function health() {
  return fetch(`${API_BASE}/health`).then(r => r.json());
}

export async function generate(prompt, num_steps = 50) {
  const res = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, num_inference_steps: num_steps }),
  });

  const text = await res.text();
  // Try to parse JSON when possible
  let payload;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (e) {
    // Not JSON — return raw text on success, otherwise throw
    if (res.ok) return { text };
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  if (!res.ok) {
    // If server returned structured error, include it
    const errDetail = payload?.detail?.error || payload?.detail || JSON.stringify(payload);
    throw new Error(`HTTP ${res.status}: ${errDetail}`);
  }

  return payload;
}

export async function uploadImage(formData) {
  return fetch(`${API_BASE}/upload`, { method: 'POST', body: formData }).then(r => r.json());
}

export async function listImages() {
  return fetch(`${API_BASE}/list`).then(r => r.json());
}
