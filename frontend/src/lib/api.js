const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  getGpuInfo: () => request("/system/gpu"),
  listModels: () => request("/system/models"),
  listTtsEngines: () => request("/system/tts-engines"),

  uploadImage: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/uploads/image`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  uploadAudio: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/uploads/audio`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  generateSingle: (params) => request("/generate", { method: "POST", body: JSON.stringify(params) }),
  generateBatch: (prompts) => request("/generate/batch", { method: "POST", body: JSON.stringify({ prompts }) }),

  listBatches: () => request("/batches"),
  getBatch: (id) => request(`/batches/${id}`),
  getJob: (id) => request(`/jobs/${id}`),

  downloadJobUrl: (id) => `${BASE}/jobs/${id}/download`,
  downloadBatchZipUrl: (id) => `${BASE}/batches/${id}/download-all`,
};
