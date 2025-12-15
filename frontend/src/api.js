const API_BASE = "https://zonal-cooperation-production.up.railway.app";

// Debug function to log CORS issues
function logRequestDetails(url, method = 'GET') {
  console.log(`📡 API Request: ${method} ${url}`);
  console.log(`📍 Origin: ${window.location.origin}`);
}

export async function health() {
  try {
    logRequestDetails(`${API_BASE}/health`);
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    console.log(`✓ Health status: ${response.status}`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("❌ Health check error:", error);
    throw error;
  }
}

export async function generate(prompt, num_steps = 50) {
  try {
    logRequestDetails(`${API_BASE}/generate`, 'POST');
    
    const payload = { 
      prompt, 
      num_inference_steps: num_steps,
      width: 1024,
      height: 1024
    };
    
    console.log("📤 Request payload:", payload);
    
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit',
    });

    console.log(`📥 Response status: ${response.status}`);

    if (!response.ok) {
      const error = await response.text();
      console.error(`❌ Generation failed (${response.status}):`, error);
      throw new Error(`Generation failed (${response.status}): ${error}`);
    }

    const data = await response.json();
    console.log("✓ Generation successful");
    
    if (data.image) {
      return { image_url: `data:image/png;base64,${data.image}` };
    }
    if (data.detail) {
      throw new Error(`Generation failed: ${data.detail}`);
    }
    throw new Error(`Unexpected response: ${JSON.stringify(data)}`);
  } catch (error) {
    console.error("❌ Generation error:", error);
    throw error;
  }
}

export async function uploadImage(formData) {
  try {
    logRequestDetails(`${API_BASE}/upload`, 'POST');
    
    const response = await fetch(`${API_BASE}/upload`, { 
      method: 'POST', 
      body: formData,
      mode: 'cors',
      credentials: 'omit',
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("✓ Upload successful");
    return data;
  } catch (error) {
    console.error("❌ Upload error:", error);
    throw error;
  }
}

export async function listImages() {
  try {
    logRequestDetails(`${API_BASE}/list`);
    
    const response = await fetch(`${API_BASE}/list`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors',
      credentials: 'omit',
    });
    
    if (!response.ok) {
      throw new Error(`List failed: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("✓ List successful");
    return data;
  } catch (error) {
    console.error("❌ List error:", error);
    throw error;
  }
}
