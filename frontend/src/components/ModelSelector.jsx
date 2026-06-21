const NOTES_SHORT = {
  "ltx-video": "Fastest · lowest VRAM",
  "wan2.1": "Best character consistency",
  "hunyuan-video": "Highest fidelity · slowest",
  svd: "Best motion control",
};

export default function ModelSelector({ models, value, onChange }) {
  if (!models?.length) {
    return <div className="text-ink-500 text-sm font-mono">loading model registry…</div>;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {models.map((m) => {
        const active = m.name === value;
        return (
          <button
            key={m.name}
            type="button"
            onClick={() => onChange(m.name)}
            className={`text-left rounded-md border px-3 py-2.5 transition-colors ${
              active
                ? "border-signal-500/60 bg-signal-500/10"
                : "border-ink-700 bg-ink-800/60 hover:border-ink-600"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className={`font-mono text-xs font-semibold ${active ? "text-signal-400" : "text-ink-200"}`}>
                {m.name}
              </span>
              <span className="text-2xs font-mono text-ink-500">{m.min_vram_gb}GB+</span>
            </div>
            <p className="text-2xs text-ink-500 mt-1 leading-snug">{NOTES_SHORT[m.name]}</p>
            <div className="flex gap-1 mt-1.5">
              {m.supports_text_to_video && <Tag>T2V</Tag>}
              {m.supports_image_to_video && <Tag>I2V</Tag>}
              {m.supports_camera_control && <Tag>CAM</Tag>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function Tag({ children }) {
  return (
    <span className="text-[0.6rem] font-mono px-1.5 py-0.5 rounded-sm bg-ink-700 text-ink-400 leading-none">
      {children}
    </span>
  );
}
