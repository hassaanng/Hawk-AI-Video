import { useEffect, useState } from "react";
import { api } from "../lib/api.js";

export default function StatusBar() {
  const [gpu, setGpu] = useState(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    const fetchGpu = () => api.getGpuInfo().then(setGpu).catch(() => setErr(true));
    fetchGpu();
    const id = setInterval(fetchGpu, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="border-b border-ink-700 bg-ink-900/80 backdrop-blur sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-sm bg-signal-500/15 border border-signal-500/40 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-signal-500 pulse-dot" />
          </div>
          <div>
            <h1 className="font-mono text-sm font-semibold tracking-tight text-white leading-none">
              AI VIDEO STUDIO
            </h1>
            <p className="text-2xs text-ink-500 font-mono leading-none mt-0.5">
              diffusion render pipeline
            </p>
          </div>
        </div>

        <div className="flex items-center gap-5 font-mono text-2xs">
          {err ? (
            <span className="text-crimson-400">GPU TELEMETRY UNAVAILABLE</span>
          ) : !gpu ? (
            <span className="text-ink-500">probing device…</span>
          ) : gpu.available ? (
            <>
              <Metric label="DEVICE" value={gpu.name} accent="text-cyan-400" />
              <Metric
                label="VRAM"
                value={`${gpu.free_vram_gb?.toFixed(1)} / ${gpu.total_vram_gb?.toFixed(1)} GB`}
                accent="text-signal-400"
              />
              <Metric label="CUDA" value={gpu.cuda_version} accent="text-ink-300" />
              <span className="flex items-center gap-1.5 text-signal-400">
                <span className="w-1.5 h-1.5 rounded-full bg-signal-500" />
                ONLINE
              </span>
            </>
          ) : (
            <span className="flex items-center gap-1.5 text-crimson-400">
              <span className="w-1.5 h-1.5 rounded-full bg-crimson-500" />
              NO CUDA DEVICE — inference disabled
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

function Metric({ label, value, accent }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-ink-500">{label}</span>
      <span className={`${accent} font-medium`}>{value || "—"}</span>
    </div>
  );
}
