import { useRef, useState } from "react";
import ModelSelector from "./ModelSelector.jsx";
import CameraMotionPicker from "./CameraMotionPicker.jsx";
import { api } from "../lib/api.js";

const RESOLUTIONS = [
  { label: "720p HD", width: 1280, height: 720 },
  { label: "768p", width: 1024, height: 768 },
  { label: "1080p", width: 1920, height: 1080 },
];

export default function GenerationForm({ models, ttsEngines, onSubmit, submitting }) {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelName, setModelName] = useState("ltx-video");
  const [mode, setMode] = useState("text-to-video");
  const [initImage, setInitImage] = useState(null);
  const [initImagePreview, setInitImagePreview] = useState(null);
  const [resolution, setResolution] = useState(RESOLUTIONS[0]);
  const [cameraMotion, setCameraMotion] = useState("static");
  const [motionStrength, setMotionStrength] = useState(0.5);
  const [enableVoiceover, setEnableVoiceover] = useState(true);
  const [narrationText, setNarrationText] = useState("");
  const [ttsEngine, setTtsEngine] = useState("xtts");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const selectedModel = models?.find((m) => m.name === modelName);
  const canImageToVideo = selectedModel?.supports_image_to_video;

  async function handleImageSelect(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setInitImagePreview(URL.createObjectURL(file));
    setUploading(true);
    try {
      const { path } = await api.uploadImage(file);
      setInitImage(path);
    } finally {
      setUploading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!prompt.trim()) return;
    onSubmit({
      prompt: prompt.trim(),
      negative_prompt: negativePrompt.trim() || null,
      model_name: modelName,
      mode: mode === "image-to-video" && initImage ? "image-to-video" : "text-to-video",
      init_image_path: mode === "image-to-video" ? initImage : null,
      width: resolution.width,
      height: resolution.height,
      num_frames: 121,
      fps: 24,
      camera_motion: cameraMotion,
      motion_strength: motionStrength,
      enable_voiceover: enableVoiceover,
      narration_text: narrationText.trim() || null,
      tts_engine: ttsEngine,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label className="block text-2xs font-mono text-ink-500 mb-2">MODEL BACKEND</label>
        <ModelSelector models={models} value={modelName} onChange={setModelName} />
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-2xs font-mono text-ink-500">GENERATION MODE</label>
          <div className="flex rounded-md border border-ink-700 overflow-hidden">
            <ModeTab active={mode === "text-to-video"} onClick={() => setMode("text-to-video")}>
              Text → Video
            </ModeTab>
            <ModeTab
              active={mode === "image-to-video"}
              disabled={!canImageToVideo}
              onClick={() => canImageToVideo && setMode("image-to-video")}
            >
              Image → Video
            </ModeTab>
          </div>
        </div>

        {mode === "image-to-video" && (
          <div className="flex items-center gap-3 mb-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-20 h-20 rounded-md border border-dashed border-ink-600 flex items-center justify-center overflow-hidden bg-ink-800 hover:border-signal-500/50 shrink-0"
            >
              {initImagePreview ? (
                <img src={initImagePreview} alt="reference" className="w-full h-full object-cover" />
              ) : (
                <span className="text-ink-500 text-2xs font-mono">+ image</span>
              )}
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
            <p className="text-2xs text-ink-500 leading-snug">
              {uploading
                ? "Uploading…"
                : initImage
                ? "Reference frame set. The model will animate motion starting from this image."
                : "Upload a still frame — it becomes the first frame of the generated clip."}
            </p>
          </div>
        )}

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the shot: subject, setting, lighting, mood, lens feel…"
          rows={4}
          className="w-full rounded-md bg-ink-800 border border-ink-700 px-3 py-2.5 text-sm text-ink-100 placeholder:text-ink-500 focus:border-signal-500/50 focus:outline-none resize-none"
        />
        <input
          value={negativePrompt}
          onChange={(e) => setNegativePrompt(e.target.value)}
          placeholder="Negative prompt (optional) — things to avoid"
          className="w-full mt-2 rounded-md bg-ink-800 border border-ink-700 px-3 py-2 text-2xs text-ink-300 placeholder:text-ink-500 focus:border-signal-500/50 focus:outline-none"
        />
      </div>

      <div>
        <label className="block text-2xs font-mono text-ink-500 mb-2">CAMERA &amp; MOTION</label>
        <CameraMotionPicker
          value={cameraMotion}
          onChange={setCameraMotion}
          strength={motionStrength}
          onStrengthChange={setMotionStrength}
        />
      </div>

      <div>
        <label className="block text-2xs font-mono text-ink-500 mb-2">OUTPUT RESOLUTION</label>
        <div className="flex gap-2">
          {RESOLUTIONS.map((r) => (
            <button
              key={r.label}
              type="button"
              onClick={() => setResolution(r)}
              className={`px-3 py-1.5 rounded-md border text-2xs font-mono ${
                resolution.label === r.label
                  ? "border-signal-500/50 bg-signal-500/10 text-signal-400"
                  : "border-ink-700 text-ink-400 hover:border-ink-600"
              }`}
            >
              {r.label}
              <span className="text-ink-500 ml-1">
                {r.width}×{r.height}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-ink-700 bg-ink-800/40 p-3.5">
        <div className="flex items-center justify-between">
          <label className="text-2xs font-mono text-ink-300 flex items-center gap-2">
            <input
              type="checkbox"
              checked={enableVoiceover}
              onChange={(e) => setEnableVoiceover(e.target.checked)}
              className="accent-signal-500"
            />
            GENERATE VOICE NARRATION
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
        {enableVoiceover && (
          <textarea
            value={narrationText}
            onChange={(e) => setNarrationText(e.target.value)}
            placeholder="Leave blank to auto-generate narration from your prompt"
            rows={2}
            className="w-full mt-2.5 rounded-md bg-ink-900 border border-ink-700 px-3 py-2 text-2xs text-ink-300 placeholder:text-ink-500 focus:border-signal-500/50 focus:outline-none resize-none"
          />
        )}
      </div>

      <button
        type="submit"
        disabled={submitting || !prompt.trim()}
        className="w-full rounded-md bg-signal-500 hover:bg-signal-400 disabled:bg-ink-700 disabled:text-ink-500 text-ink-950 font-mono text-sm font-semibold py-2.5 transition-colors"
      >
        {submitting ? "QUEUEING…" : "RENDER CLIP"}
      </button>
    </form>
  );
}

function ModeTab({ active, disabled, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-1 text-2xs font-mono transition-colors ${
        active
          ? "bg-signal-500/15 text-signal-400"
          : disabled
          ? "text-ink-600 cursor-not-allowed"
          : "text-ink-400 hover:bg-ink-700/50"
      }`}
    >
      {children}
    </button>
  );
}
