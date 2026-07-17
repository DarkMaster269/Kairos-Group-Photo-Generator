/**
 * api.js — Kairos API service layer
 *
 * All network calls in one place. Every function returns a plain object
 * or throws a descriptive Error so callers can handle network vs. server
 * errors without digging into raw fetch responses.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

/** Generic JSON fetcher — throws on non-2xx or network failure */
async function apiFetch(path, init = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      ...init,
    });
  } catch (_networkErr) {
    throw new Error(
      `Cannot reach the Kairos server at ${API_BASE}. ` +
      'Make sure the FastAPI backend is running (uvicorn app.main:app --reload).'
    );
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore parse error */ }
    throw new Error(detail);
  }

  return response.json();
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * POST /api/burst
 * @param {File[]} files — 5–15 image files to upload
 * @returns {{ burst_id: string, photo_count: number, uploaded_at: string }}
 */
export async function uploadBurst(files) {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  return apiFetch('/api/burst', { method: 'POST', body: formData });
}

/**
 * GET /api/burst/{id}/status
 * @returns {{ burst_id, status, progress_percentage, message }}
 */
export async function getBurstStatus(burstId) {
  return apiFetch(`/api/burst/${burstId}/status`);
}

/**
 * GET /api/burst/{id}/result
 * @returns {PipelineResult} — final result including output_image_url, gate results, etc.
 */
export async function getBurstResult(burstId) {
  return apiFetch(`/api/burst/${burstId}/result`);
}

/**
 * POST /api/burst/{id}/retry
 * @returns {StatusResponse}
 */
export async function retryBurst(burstId) {
  return apiFetch(`/api/burst/${burstId}/retry`, { method: 'POST' });
}
