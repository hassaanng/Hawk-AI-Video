import { useState } from "react";
import { api } from "../lib/api.js";

const STATUS_STYLES = {
  queued: { color: "text-ink-400", dot: "bg-ink-500", label: "QUEUED" },
  running: { color: "text-amber-400", dot: "bg-amber-400", label: "RENDERING" },
  done: { color: "text-signal-400", dot: "bg-signal-500", label: "DONE" },
  failed: { color: "text-crimson-400", dot: "bg-crimson-500", label: "FAILED" },
  canceled: { color: "text-ink-500", dot: "bg-ink-600", label: "CANCELED" },
};

export default function JobQueueList({ batch }) {
  const [expandedError, setExpandedError] = useState(null);
  const [previewJobId, setPreviewJobId] = useState(null);

  if (!batch) return null;

  const { jobs, status, zip_path, id: batchId } = batch;
  const doneCount = jobs.filter((j) => j.status === "done").length;
  const failedCount = jobs.filter((j) => j.status === "failed").length;
  const canDownloadAll = !!zip_path;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BatchStatusBadge status={status} />
          <span className="text-2xs font-mono text-ink-500">
            {doneCount}/{jobs.length} complete{failedCount > 0 && ` · ${failedCount} failed`}
          </span>
        </div>
        {jobs.length > 1 && (
          <a
            href={canDownloadAll ? api.downloadBatchZipUrl(batchId) : undefined}
            className={`text-2xs font-mono px-3 py-1.5 rounded-md border transition-colors ${
              canDownloadAll
                ? "border-signal-500/50 text-signal-400 hover:bg-signal-500/10"
                : "border-ink-700 text-ink-600 cursor-not-allowed"
            }`}
            onClick={(e) => !canDownloadAll && e.preventDefault()}
          >
            ⇓ DOWNLOAD ALL (.zip)
          </a>
        )}
      </div>

      <div className="rounded-md border border-ink-700 divide-y divide-ink-700 overflow-hidden">
        {jobs.map((job) => {
          const style = STATUS_STYLES[job.status] || STATUS_STYLES.queued;
          const isPreviewing = previewJobId === job.id;

          return (
            <div key={job.id} className="bg-ink-800/40">
              <div className="px-3.5 py-3 flex items-center gap-3">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${style.dot} ${job.status === "running" ? "pulse-dot" : ""}`} />

                <div className="flex-1 min-w-0">
                  <p className="text-sm text-ink-200 truncate">{job.prompt}</p>
                  {job.status === "running" && (
                    <div className="mt-1.5 flex items-center gap-2">
                      <div className="flex-1 h-1 rounded-full bg-ink-700 overflow-hidden relative">
                        <div
                          className="h-full bg-amber-400 transition-all duration-500"
                          style={{ width: `${Math.round(job.progress * 100)}%` }}
                        />
                      </div>
                      <span className="text-2xs font-mono text-ink-500 w-9 text-right">
                        {Math.round(job.progress * 100)}%
                      </span>
                    </div>
                  )}
                  {job.status === "running" && job.status_message && (
                    <p className="text-2xs font-mono text-ink-500 mt-0.5">{job.status_message}</p>
                  )}
                  {job.status === "failed" && (
                    <button
                      type="button"
                      onClick={() => setExpandedError(expandedError === job.id ? null : job.id)}
                      className="text-2xs font-mono text-crimson-400 hover:underline mt-0.5"
                    >
                      {expandedError === job.id ? "hide error" : "show error"}
                    </button>
                  )}
                </div>

                <span className={`text-2xs font-mono shrink-0 ${style.color}`}>{style.label}</span>

                {job.status === "done" && (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      type="button"
                      onClick={() => setPreviewJobId(isPreviewing ? null : job.id)}
                      className="text-2xs font-mono px-2.5 py-1 rounded-md border border-ink-600 text-ink-300 hover:border-ink-500"
                    >
                      {isPreviewing ? "hide" : "preview"}
                    </button>
                    <a
                      href={api.downloadJobUrl(job.id)}
                      className="text-2xs font-mono px-2.5 py-1 rounded-md border border-signal-500/50 text-signal-400 hover:bg-signal-500/10"
                    >
                      ⇓ download
                    </a>
                  </div>
                )}
              </div>

              {expandedError === job.id && job.error && (
                <div className="px-3.5 pb-3 -mt-1">
                  <pre className="text-2xs font-mono text-crimson-300/80 bg-ink-950 rounded-md p-2.5 overflow-x-auto whitespace-pre-wrap">
                    {job.error}
                  </pre>
                </div>
              )}

              {isPreviewing && (
                <div className="px-3.5 pb-3.5">
                  <video
                    src={api.downloadJobUrl(job.id)}
                    controls
                    autoPlay
                    className="w-full rounded-md border border-ink-700 bg-black max-h-96"
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BatchStatusBadge({ status }) {
  const style = STATUS_STYLES[status] || { color: "text-ink-400", dot: "bg-ink-500", label: status?.toUpperCase() };
  const isPartial = status === "partial";
  return (
    <span
      className={`flex items-center gap-1.5 text-2xs font-mono px-2 py-1 rounded-md border ${
        isPartial ? "border-amber-400/40 text-amber-400" : "border-ink-700 " + style.color
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${isPartial ? "bg-amber-400" : style.dot}`} />
      BATCH {isPartial ? "PARTIAL" : style.label}
    </span>
  );
}
