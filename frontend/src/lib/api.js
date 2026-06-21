// This file's BASE URL is no longer hardcoded to "/api". Instead, it reads
// the current backend address from a GitHub Gist that the Kaggle notebook
// updates automatically every time a new Cloudflare tunnel is created. This
// means the frontend (deployed once on Cloudflare Pages) never needs to be
// redeployed when the Kaggle backend gets a new tunnel URL after a restart.

const GIST_RAW_URL =
  "https://gist.githubusercontent.com/hassaanng/7bb66b71bd373f26f17c6bf666680137/raw/backend_url.txt";

let BASE = "/api"; // fallback used only until the Gist fetch resolves

async function loadBackendUrl() {
  try {
    // ?t=Date.now() busts any caching so we always get the latest address,
    // not a stale cached copy of the Gist's raw file.
    const res = await fetch(`${GIST_RAW_URL}?t=${Date.now()}`);
    const url = (await res.text()).trim();
    if (url.startsWith("https://")) {
      BASE = `${url}/api`;
      console.log("Backend URL loaded from Gist:", BASE);
    } else {
      console.warn("Gist content did not look like a URL, using fallback:", url);
    }
  } catch (e) {
    console.error("Could not load backend URL from Gist, using fallback /api", e);
  }
}

// Kick off the fetch as soon as this module loads. Every API call below
// awaits this same promise, so the very first call in the app also waits
// for the real backend address instead of racing ahead with "/api".
const backendUrlPromise = loadBackendUrl();

async function request(path, options = {}) {
  await backendUrlPromise;
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
    await backendUrlPromise;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/uploads/image`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  uploadAudio: async (file) => {
    await backendUrlPromise;
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

  // These build plain URLs (not via request()) because they're used directly
  // as href/src attributes in the UI, not fetched via JS - so they need BASE
  // to already be resolved. We expose a helper that components can await.
  downloadJobUrl: (id) => `${BASE}/jobs/${id}/download`,
  downloadBatchZipUrl: (id) => `${BASE}/batches/${id}/download-all`,

  // Components that render <a href> or <video src> before the Gist fetch
  // resolves should call this first to guarantee BASE is ready.
  ensureBackendUrlLoaded: () => backendUrlPromise,
};
