const OPTIONS = [
  { id: "static", label: "Static", glyph: "▣" },
  { id: "pan_left", label: "Pan L", glyph: "⟲" },
  { id: "pan_right", label: "Pan R", glyph: "⟳" },
  { id: "zoom_in", label: "Zoom In", glyph: "⊕" },
  { id: "zoom_out", label: "Zoom Out", glyph: "⊖" },
  { id: "orbit", label: "Orbit", glyph: "◎" },
  { id: "dolly_in", label: "Dolly In", glyph: "→" },
];

export default function CameraMotionPicker({ value, onChange, strength, onStrengthChange }) {
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {OPTIONS.map((o) => (
          <button
            key={o.id}
            type="button"
            onClick={() => onChange(o.id)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-2xs font-mono transition-colors ${
              value === o.id
                ? "border-cyan-400/50 bg-cyan-400/10 text-cyan-400"
                : "border-ink-700 text-ink-400 hover:border-ink-600"
            }`}
          >
            <span className="text-sm leading-none">{o.glyph}</span>
            {o.label}
          </button>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-3">
        <span className="text-2xs font-mono text-ink-500 w-24">MOTION {Math.round(strength * 100)}%</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={strength}
          onChange={(e) => onStrengthChange(parseFloat(e.target.value))}
          className="flex-1 accent-signal-500"
        />
      </div>
    </div>
  );
}
