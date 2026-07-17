/**
 * usePipeline.js — Custom React hook owning the full pipeline state machine.
 *
 * Exposes:
 *   state   : { stage, progress, message, result, error, burstId, filePreviews }
 *   actions : upload(files), retry(), reset()
 *
 * Pipeline stages (frontend-side):
 *   idle → uploading → gate1 → detecting → blending → gate2 → retrying → done/fallback/error
 *
 * Progress-to-stage mapping (backend sends progress_percentage 0-100):
 *   0–15   → gate1
 *   16–55  → detecting
 *   56–72  → blending
 *   73–92  → gate2
 *   93–99  → retrying  (only if retry_count > 0)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { uploadBurst, getBurstStatus, getBurstResult, retryBurst } from '../services/api';

// ── Stage constants ──────────────────────────────────────────────────────────
export const STAGE = {
  IDLE:       'idle',
  UPLOADING:  'uploading',
  GATE1:      'gate1',
  DETECTING:  'detecting',
  BLENDING:   'blending',
  GATE2:      'gate2',
  RETRYING:   'retrying',
  DONE:       'done',
  FALLBACK:   'fallback',
  ERROR:      'error',
};

/** Maps backend status + progress to a frontend stage label */
function deriveStage(backendStatus, progress) {
  if (backendStatus === 'complete')  return STAGE.DONE;
  if (backendStatus === 'fallback')  return STAGE.FALLBACK;
  if (backendStatus === 'error')     return STAGE.ERROR;
  if (progress <= 15)                return STAGE.GATE1;
  if (progress <= 55)                return STAGE.DETECTING;
  if (progress <= 72)                return STAGE.BLENDING;
  if (progress <= 92)                return STAGE.GATE2;
  return STAGE.RETRYING;
}

const POLL_INTERVAL_MS = 1400;

// ── Hook ─────────────────────────────────────────────────────────────────────
export function usePipeline() {
  const [stage,        setStage]        = useState(STAGE.IDLE);
  const [progress,     setProgress]     = useState(0);
  const [message,      setMessage]      = useState('');
  const [result,       setResult]       = useState(null);
  const [error,        setError]        = useState(null);   // { type: 'gate1'|'gate2'|'network', detail: string, issues: [] }
  const [burstId,      setBurstId]      = useState(null);
  const [filePreviews, setFilePreviews] = useState([]);

  const burstIdRef = useRef(null);
  const pollingRef = useRef(null);

  // Keep ref in sync so the polling closure always reads the current burstId
  useEffect(() => { burstIdRef.current = burstId; }, [burstId]);

  // ── Stop polling helper ───────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  // ── Fetch final result ────────────────────────────────────────────────────
  const fetchResult = useCallback(async (id) => {
    try {
      const data = await getBurstResult(id);
      setResult(data);
      setProgress(100);
      setMessage('Verification complete — composite generated.');

      if (data.result_type === 'fallback_single_frame') {
        setStage(STAGE.FALLBACK);
      } else if (data.gate_2_result && !data.gate_2_result.passed) {
        // Gate 2 explicitly flagged — surface it as an error (pipeline still returns result)
        setError({
          type: 'gate2',
          detail: 'Gate 2 flagged blending issues in the composite.',
          issues: data.gate_2_result.issues || [],
        });
        setStage(STAGE.DONE);
      } else {
        setStage(STAGE.DONE);
      }
    } catch (err) {
      stopPolling();
      setStage(STAGE.ERROR);
      setError({ type: 'network', detail: err.message, issues: [] });
    }
  }, [stopPolling]);

  // ── Start polling ─────────────────────────────────────────────────────────
  const startPolling = useCallback((id) => {
    stopPolling();

    pollingRef.current = setInterval(async () => {
      try {
        const status = await getBurstStatus(id);
        const newStage = deriveStage(status.status, status.progress_percentage);

        setProgress(status.progress_percentage);
        setMessage(status.message);
        setStage(newStage);

        if (status.status === 'complete' || status.status === 'fallback') {
          stopPolling();
          await fetchResult(id);
        } else if (status.status === 'error') {
          stopPolling();
          setStage(STAGE.ERROR);
          setError({ type: 'pipeline', detail: status.message, issues: [] });
        }
      } catch (err) {
        // Network error during polling — stop so we don't spam errors
        stopPolling();
        setStage(STAGE.ERROR);
        setError({ type: 'network', detail: err.message, issues: [] });
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling, fetchResult]);

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  // ── upload action ─────────────────────────────────────────────────────────
  const upload = useCallback(async (files) => {
    if (!files || files.length === 0) return;

    // Generate preview URLs
    const previews = Array.from(files).map(f => URL.createObjectURL(f));
    setFilePreviews(previews);

    // Reset state
    setStage(STAGE.UPLOADING);
    setProgress(3);
    setMessage('Uploading burst photos…');
    setResult(null);
    setError(null);
    setBurstId(null);

    try {
      const { burst_id } = await uploadBurst(Array.from(files));
      setBurstId(burst_id);
      burstIdRef.current = burst_id;
      setStage(STAGE.GATE1);
      setProgress(12);
      setMessage('Validating input quality (Gate 1)…');
      startPolling(burst_id);
    } catch (err) {
      setStage(STAGE.ERROR);
      setError({
        type: 'network',
        detail: err.message,
        issues: [],
      });
    }
  }, [startPolling]);

  // ── retry action ──────────────────────────────────────────────────────────
  const retry = useCallback(async () => {
    const id = burstIdRef.current;
    if (!id) return;

    setStage(STAGE.GATE2);
    setProgress(70);
    setMessage('Manual retry triggered — re-evaluating blend…');
    setError(null);

    try {
      await retryBurst(id);
      startPolling(id);
    } catch (err) {
      setStage(STAGE.ERROR);
      setError({ type: 'network', detail: err.message, issues: [] });
    }
  }, [startPolling]);

  // ── reset action ──────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    stopPolling();
    setStage(STAGE.IDLE);
    setProgress(0);
    setMessage('');
    setResult(null);
    setError(null);
    setBurstId(null);
    setFilePreviews([]);
  }, [stopPolling]);

  return {
    stage, progress, message, result, error, burstId, filePreviews,
    upload, retry, reset,
  };
}
