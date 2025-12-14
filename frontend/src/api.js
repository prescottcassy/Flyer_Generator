const API_BASE = "https://zonal-cooperation-production.up.railway.app";

export async function health() {
  return fetch(`${API_BASE}/health`).then(r => r.json());
}

export async function generate(prompt, num_steps = 50) {
  const res = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      prompt, 
      num_inference_steps: num_steps,
      width: 1024,
      height: 1024
    }),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`Generation failed (${res.status}): ${error}`);
  }

  const data = await res.json();
  if (data.image) {
    return { image_url: `data:image/png;base64,${data.image}` };
  }
  if (data.detail) {
    throw new Error(`Generation failed: ${data.detail}`);
  }
  throw new Error(`Unexpected response: ${JSON.stringify(data)}`);
}

export async function uploadImage(formData) {
  return fetch(`${API_BASE}/upload`, { method: 'POST', body: formData }).then(r => r.json());
}

export async function listImages() {
  return fetch(`${API_BASE}/list`).then(r => r.json());
}
