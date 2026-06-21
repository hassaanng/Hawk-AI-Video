import { useRef, useState } from "react";

const MAX_PROMPTS = 50;

export default function BatchUploadPanel({ models, ttsEngines, onSubmit, submitting }) {
  const [rawText, setRawText] = useState("");
  const [modelName, setModelName] = useState("ltx-video");
  const [enableVoiceover, setEnableVoiceover] = useState(true);
  const [ttsEngine, setTtsEngine] = useState("xtts");
  const fileInputRef = useRef(null);

  const lines = rawText
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const overLimit = lines.length > MAX_PROMPTS;

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setRawText(text);
  }

  function handleSubmit() {
    if (!lines.length || overLimit) return;
    const prompts = lines.slice(0, MAX_PROMPTS).map((prompt) => ({
      prompt,
      model_name: modelName,
      width: 1280,
      height: 720,
      num_frames: 121,
      fps: 24,
      camera_motion: "static",
      motion_strength: 0.5,
      enable_voiceover: enableVoiceover,
      tts_engine: ttsEngine,
    }));
    onSubmit(prompts);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-2xs font-mono text-ink-500">
          ONE PROMPT PER LINE — UP TO {MAX_PROMPTS}
        </label>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="text-2xs font-mono text-cyan-400 hover:text-cyan-300"
        >
          upload .txt file
        </button>
        <input ref={fileInputRef} type="file" accept=".txt" onChange={handleFile} className="hidden" />
      </div>

      <textarea
        value={rawText}
        onChange={(e) => setRawText(e.target.value)}
        placeholder={"a hawk gliding over a misty canyon at dawn\na bustling night market in tokyo, neon reflections in rain puddles\n..."}
        rows={10}
        className="w-full rounded-md bg-ink-800 border border-ink-700 px-3 py-2.5 text-sm text-ink-100 placeholder:text-ink-500 focus:border-signal-500/50 focus:outline-none resize-none font-mono"
      />

      <div className="flex items-center justify-between text-2xs font-mono">
        <span className={overLimit ? "text-crimson-400" : "text-ink-500"}>
          {lines.length} / {MAX_PROMPTS} prompts {overLimit && "— trim to fit"}
        </span>
      </div>

      <div>
        <label className="block text-2xs font-mono text-ink-500 mb-2">MODEL FOR ALL CLIPS IN THIS BATCH</label>
        <div className="flex gap-1.5 flex-wrap">
          {(models || []).map((m) => (
            <button
              key={m.name}
              type="button"
              onClick={() => setModelName(m.name)}
              className={`px-2.5 py-1 rounded-md border text-2xs font-mono ${
                modelName === m.name
                  ? "border-signal-500/50 bg-signal-500/10 text-signal-400"
                  : "border-ink-700 text-ink-400 hover:border-ink-600"
              }`}
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-ink-700 bg-ink-800/40 p-3.5 flex items-center justify-between">
        <label className="text-2xs font-mono text-ink-300 flex items-center gap-2">
          <input
            type="checkbox"
            checked={enableVoiceover}
            onChange={(e) => setEnableVoiceover(e.target.checked)}
            className="accent-signal-500"
          />
          AUTO-NARRATE EVERY CLIP
        </label>
        {enableVoiceover && (
          <select
            value={ttsEngine}
            onChange={(e) => setTtsEngine(e.target.value)}
            className="bg-ink-800 border border-ink-700 rounded-md text-2xs font-mono text-ink-300 px-2 py-1 focus:outline-none focus:border-signal-500/50"
          >
            {(ttsEngines || ["xtts", "piper"]).map((eng) => (
              <option key={eng} value={eng}>
                {eng}
              </option>
            ))}
          </select>
        )}
      </div>

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitting || !lines.length || overLimit}
        className="w-full rounded-md bg-signal-500 hover:bg-signal-400 disabled:bg-ink-700 disabled:text-ink-500 text-ink-950 font-mono text-sm font-semibold py-2.5 transition-colors"
      >
        {submitting ? "QUEUEING BATCH…" : `QUEUE ${lines.length || ""} CLIP${lines.length === 1 ? "" : "S"}`}
      </button>
    </div>
  );
}
