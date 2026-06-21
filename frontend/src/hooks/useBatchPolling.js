import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";

const TERMINAL_STATES = new Set(["done", "failed", "partial", "canceled"]);

export function useBatchPolling(batchId) {
  const [batch, setBatch] = useState(null);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!batchId) {
      setBatch(null);
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const data = await api.getBatch(batchId);
        if (cancelled) return;
        setBatch(data);
        setError(null);
        if (TERMINAL_STATES.has(data.status) && intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 1500);

    return () => {
      cancelled = true;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [batchId]);

  return { batch, error };
}
