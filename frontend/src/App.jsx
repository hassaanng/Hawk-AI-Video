import { useEffect, useState } from "react";
import StatusBar from "./components/StatusBar.jsx";
import GenerationForm from "./components/GenerationForm.jsx";
import BatchUploadPanel from "./components/BatchUploadPanel.jsx";
import JobQueueList from "./components/JobQueueList.jsx";
import { useBatchPolling } from "./hooks/useBatchPolling.js";
import { api } from "./lib/api.js";

export default function App() {
  const [tab, setTab] = useState("single");
  const [models, setModels] = useState([]);
  const [ttsEngines, setTtsEngines] = useState([]);
  const [activeBatchId, setActiveBatchId] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const { batch, error: pollError } = useBatchPolling(activeBatchId);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => {});
    api.listTtsEngines().then((r) => setTtsEngines(r.engines)).catch(() => {});
  }, []);

  async function handleSingleSubmit(params) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await api.generateSingle(params);
      setActiveBatchId(result.id);
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleBatchSubmit(prompts) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await api.generateBatch(prompts);
      setActiveBatchId(result.id);
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen">
      <StatusBar />

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[1fr_1.1fr] gap-8">
        <section>
          <div className="flex items-center gap-1 mb-5 border border-ink-700 rounded-md p-1 w-fit">
            <TabButton active={tab === "single"} onClick={() => setTab("single")}>
              Single Clip
            </TabButton>
            <TabButton active={tab === "batch"} onClick={() => setTab("batch")}>
              Batch (up to 50)
            </TabButton>
          </div>

          {submitError && (
            <div className="mb-4 rounded-md border border-crimson-500/40 bg-crimson-500/10 px-3.5 py-2.5 text-sm text-crimson-300">
              {submitError}
            </div>
          )}

          {tab === "single" ? (
            <GenerationForm models={models} ttsEngines={ttsEngines} onSubmit={handleSingleSubmit} submitting={submitting} />
          ) : (
            <BatchUploadPanel models={models} ttsEngines={ttsEngines} onSubmit={handleBatchSubmit} submitting={submitting} />
          )}
        </section>

        <section>
          <h2 className="text-2xs font-mono text-ink-500 mb-3">RENDER QUEUE</h2>
          {pollError && <p className="text-crimson-400 text-sm">{pollError}</p>}
          {!batch ? (
            <div className="rounded-md border border-dashed border-ink-700 px-5 py-10 text-center">
              <p className="text-ink-500 text-sm">No active render. Submit a prompt to start.</p>
            </div>
          ) : (
            <JobQueueList batch={batch} />
          )}
        </section>
      </main>

      <footer className="max-w-7xl mx-auto px-6 pb-8">
        <p className="text-2xs font-mono text-ink-600">
          AI Video Studio — local/RunPod diffusion pipeline. Backends: LTX-Video · Wan 2.1 · HunyuanVideo · SVD.
        </p>
      </footer>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3.5 py-1.5 rounded-sm text-2xs font-mono transition-colors ${
        active ? "bg-signal-500/15 text-signal-400" : "text-ink-400 hover:text-ink-200"
      }`}
    >
      {children}
    </button>
  );
}
