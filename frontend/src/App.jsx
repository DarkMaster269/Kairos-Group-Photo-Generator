import { useRef, useState, useCallback } from 'react';
import {
  Camera, Upload, CheckCircle, XCircle, Loader2,
  RefreshCw, AlertTriangle, ChevronLeft, ChevronRight,
  Download, RotateCcw, Sparkles,
} from 'lucide-react';
import { usePipeline, STAGE } from './hooks/usePipeline';

// ── Pipeline stage config (drives the tracker UI) ───────────────────────────
const PIPELINE_STEPS = [
  { id: STAGE.GATE1,     label: 'Gate 1 — Input Check',          desc: 'Validating scene, quality, face count' },
  { id: STAGE.DETECTING, label: 'Face Detection & Clustering',    desc: 'Finding & grouping identities across frames' },
  { id: STAGE.BLENDING,  label: 'Expression Score & Blend',       desc: 'Selecting best faces · Poisson compositing' },
  { id: STAGE.GATE2,     label: 'Gate 2 — Output Verification',   desc: 'AI checks composite for seams & mismatches' },
  { id: STAGE.RETRYING,  label: 'Retry / Fallback',               desc: 'Swapping faces or falling back to best frame' },
];

const STAGE_ORDER = [STAGE.GATE1, STAGE.DETECTING, STAGE.BLENDING, STAGE.GATE2, STAGE.RETRYING];

function stepStatus(stepId, currentStage, overallStage) {
  const stepIdx    = STAGE_ORDER.indexOf(stepId);
  const currentIdx = STAGE_ORDER.indexOf(currentStage);
  if (overallStage === STAGE.ERROR && stepIdx === currentIdx) return 'error';
  if (stepIdx < currentIdx)  return 'done';
  if (stepIdx === currentIdx) return 'active';
  if (overallStage === STAGE.DONE || overallStage === STAGE.FALLBACK) {
    return stepIdx <= STAGE_ORDER.indexOf(STAGE.RETRYING) ? 'done' : 'pending';
  }
  return 'pending';
}

// ── Drag-to-reveal comparison slider ────────────────────────────────────────
function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);
  const containerRef = useRef(null);

  const move = useCallback((clientX) => {
    if (!containerRef.current) return;
    const { left, width } = containerRef.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - left) / width) * 100)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-[4/3] overflow-hidden cursor-ew-resize select-none rounded-sm shadow-[0_20px_40px_rgba(20,15,10,0.25)]"
      onMouseDown={() => { dragging.current = true; }}
      onMouseMove={e => { if (dragging.current) move(e.clientX); }}
      onMouseUp={() => { dragging.current = false; }}
      onMouseLeave={() => { dragging.current = false; }}
      onTouchMove={e => move(e.touches[0].clientX)}
      role="slider"
      aria-label="Before/After comparison slider"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      {/* BEFORE — base frame */}
      <div className="absolute inset-0 bg-[#1b1712]">
        <img src={before} alt="original base frame" className="w-full h-full object-cover opacity-80" />
        <span className="absolute top-3 left-3 font-mono text-[10px] tracking-widest bg-black/50 text-white/70 px-2 py-0.5">BEFORE</span>
      </div>

      {/* AFTER — composite, clipped left of divider */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img src={after} alt="optimized composite" className="w-full h-full object-cover" />
        <span className="absolute top-3 left-3 font-mono text-[10px] tracking-widest bg-black/50 text-white/70 px-2 py-0.5">AFTER</span>
      </div>

      {/* Divider line + handle */}
      <div
        className="absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_8px_rgba(255,255,255,0.6)]"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-[#F7F1E1] shadow-lg border border-black/10 flex items-center justify-center gap-0.5">
          <ChevronLeft className="w-3.5 h-3.5 text-[#1A1610]" />
          <ChevronRight className="w-3.5 h-3.5 text-[#1A1610]" />
        </div>
      </div>
    </div>
  );
}

// ── Step icon based on status ────────────────────────────────────────────────
function StepIcon({ status }) {
  if (status === 'done')   return <CheckCircle className="w-5 h-5 text-[#4B7A51] flex-shrink-0" />;
  if (status === 'error')  return <XCircle     className="w-5 h-5 text-[#B23A2E] flex-shrink-0" />;
  if (status === 'active') return <Loader2     className="w-5 h-5 text-[#2F7A78] flex-shrink-0 animate-spin" />;
  return <div className="w-5 h-5 rounded-full border-2 border-[#1A1610]/20 flex-shrink-0" />;
}

// ── Error warning card ───────────────────────────────────────────────────────
function ErrorCard({ error, onRetry, onReset }) {
  const isNetwork = error?.type === 'network';
  const isGate1   = error?.type === 'gate1';

  return (
    <div className="my-8 p-5 bg-[#F7F1E1] border-l-4 border-[#B23A2E] shadow-[0_8px_20px_rgba(20,15,10,0.15)] max-w-[480px]">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-[#B23A2E] flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-bricolage font-bold text-[#1A1610] text-[15px] mb-1">
            {isNetwork ? 'Server unreachable' : isGate1 ? 'Burst rejected — Gate 1' : 'Pipeline error'}
          </p>
          <p className="font-mono text-[12px] text-[#6b6354] mb-3">{error?.detail}</p>

          {isNetwork && (
            <p className="font-mono text-[11px] text-[#6b6354] bg-black/5 p-2 mb-3">
              → Start backend: <code>uvicorn app.main:app --reload</code> in <code>/backend</code>
            </p>
          )}

          {error?.issues?.length > 0 && (
            <ul className="font-mono text-[11px] text-[#6b6354] list-disc list-inside mb-3 space-y-0.5">
              {error.issues.map((iss, i) => (
                <li key={i}>[{iss.issue_type}] {iss.description}</li>
              ))}
            </ul>
          )}

          <div className="flex gap-2 flex-wrap">
            {onRetry && (
              <button
                onClick={onRetry}
                className="flex items-center gap-1.5 font-mono text-[11px] bg-[#2F7A78] hover:bg-[#256260] text-white px-3 py-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F7A78]"
              >
                <RefreshCw className="w-3 h-3" /> Retry blend
              </button>
            )}
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 font-mono text-[11px] bg-[#1A1610]/10 hover:bg-[#1A1610]/20 text-[#1A1610] px-3 py-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1A1610]"
            >
              <RotateCcw className="w-3 h-3" /> Start over
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const { stage, progress, message, result, error, burstId, filePreviews, upload, retry, reset } = usePipeline();

  const fileInputRef = useRef(null);
  const [fileError, setFileError] = useState('');

  const isProcessing = ![STAGE.IDLE, STAGE.DONE, STAGE.FALLBACK, STAGE.ERROR].includes(stage);
  const isComplete   = stage === STAGE.DONE || stage === STAGE.FALLBACK;

  // ── File selection ─────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length < 5 || files.length > 15) {
      setFileError(`Select between 5 and 15 photos (got ${files.length}).`);
      return;
    }
    setFileError('');
    upload(files);
  };

  // ── Drag and drop ──────────────────────────────────────────────────────────
  const [draggingOver, setDraggingOver] = useState(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDraggingOver(false);
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (files.length < 5 || files.length > 15) {
      setFileError(`Drop between 5 and 15 images (got ${files.length}).`);
      return;
    }
    setFileError('');
    upload(files);
  };

  return (
    <div className="max-w-[1040px] mx-auto px-5 pb-24 relative">

      {/* ── Fixed frame counter nav ── */}
      <nav
        className="fixed top-[22px] right-[22px] z-50 bg-[#1B1611] px-[14px] py-[10px] shadow-[0_10px_20px_rgba(0,0,0,0.28)] hidden sm:flex items-center gap-3"
        aria-hidden="true"
      >
        <div className="flex flex-col leading-tight">
          <span className="font-mono text-[#E7EAE2] text-[15px] font-bold">
            {isComplete ? '05' : isProcessing ? '03' : '01'}
          </span>
          <span className="font-mono text-[rgba(231,234,226,0.5)] text-[8px] uppercase tracking-widest">
            {isComplete ? 'result' : isProcessing ? 'check' : 'shot'}
          </span>
        </div>
        <div className="flex flex-col gap-[5px]">
          {['s-hero','s-pipeline','s-result'].map((id, i) => (
            <button
              key={id}
              onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })}
              className={`h-[3px] border-none p-0 cursor-pointer transition-all duration-300 ${
                (i === 0 && stage === STAGE.IDLE) || (i === 1 && isProcessing) || (i === 2 && isComplete)
                  ? 'w-[26px] bg-[#2F7A78]'
                  : 'w-[20px] bg-[rgba(231,234,226,0.25)]'
              }`}
              aria-label={`Jump to section ${i + 1}`}
            />
          ))}
        </div>
      </nav>

      {/* ── Ticket header ── */}
      <div
        className="inline-block bg-[#F7F1E1] shadow-[0_6px_16px_rgba(20,15,10,0.18)] px-[18px] py-[10px] mt-8 -rotate-[1.5deg] font-bricolage font-extrabold text-[19px] relative
          before:content-[''] before:absolute before:top-[-6px] before:left-[14px] before:w-[13px] before:h-[13px] before:rounded-full before:bg-[#9a958a] before:shadow-[0_2px_3px_rgba(0,0,0,0.3)]"
      >
        KAIROS
        <small className="block font-mono font-normal text-[10px] tracking-widest text-[#6b6354] mt-[2px]">
          {isComplete ? `roll 01 · ${result?.result_type === 'blended' ? 'composite' : 'fallback'}` : 'roll 01 · waiting'}
        </small>
      </div>

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 1 — Hero + Upload
      ══════════════════════════════════════════════════════════════════════ */}
      <section id="s-hero" className="flex items-end gap-[60px] flex-wrap py-[60px]">

        {/* Polaroid preview */}
        <div className="bg-[#F7F1E1] p-[14px_14px_20px] w-[260px] shadow-[0_18px_34px_rgba(20,15,10,0.20)] -rotate-[2.5deg] flex-shrink-0 relative
          before:content-[''] before:absolute before:top-[-16px] before:left-1/2 before:-translate-x-1/2 before:w-[60px] before:h-[19px] before:bg-[rgba(200,155,60,0.4)] before:-rotate-[1deg] before:shadow-[0_2px_4px_rgba(0,0,0,0.12)]"
        >
          <div className="aspect-square bg-gradient-to-br from-[#2a2521] to-[#1b1712] flex items-center justify-center overflow-hidden">
            {filePreviews.length > 0
              ? <img src={filePreviews[Math.floor(filePreviews.length / 2)]} alt="preview" className="w-full h-full object-cover" />
              : <Camera className="w-12 h-12 text-[#E7EAE2] opacity-30" />
            }
          </div>
          <div className="font-hand text-[21px] text-[#1A1610] text-center mt-2 leading-tight">
            {isComplete ? 'the keeper.' : 'waiting for\nyour burst'}
          </div>
        </div>

        {/* Copy + upload */}
        <div className="flex-1 min-w-[280px] pb-2">
          <h1 className="font-bricolage font-extrabold text-[clamp(34px,5.5vw,52px)] leading-[1.02] text-[#1A1610]">
            Nobody blinks<br />in <span className="text-[#B23A2E]">the keeper.</span>
          </h1>
          <p className="max-w-[44ch] text-[#6b6354] text-[15px] mt-4 mb-6">
            Drop a burst of 5–15 group photos. Kairos scores every face, composites the best expressions into one seamless photo, and double-checks the result before you ever see it.
          </p>

          {/* Drop zone */}
          {stage === STAGE.IDLE && (
            <div
              onDragOver={e => { e.preventDefault(); setDraggingOver(true); }}
              onDragLeave={() => setDraggingOver(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-sm p-6 text-center transition-all ${
                draggingOver
                  ? 'border-[#2F7A78] bg-[#2F7A78]/5'
                  : 'border-[#1A1610]/20 hover:border-[#1A1610]/40'
              }`}
            >
              <Upload className="mx-auto mb-2 w-8 h-8 text-[#1A1610]/30" />
              <p className="font-mono text-[12px] text-[#6b6354] mb-3">
                drag photos here, or
              </p>
              <label
                htmlFor="burst-upload"
                className="inline-block cursor-pointer bg-[#1B1611] hover:bg-[#2A2521] text-[#E7EAE2] font-mono text-[12px] tracking-wide px-4 py-2 transition-colors
                  focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[#2F7A78]"
              >
                <span className="flex items-center gap-2">
                  <Camera className="w-3.5 h-3.5" /> Insert burst
                </span>
                <input
                  id="burst-upload"
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileChange}
                  className="sr-only"
                />
              </label>
              <p className="font-mono text-[10px] text-[#6b6354]/60 mt-2">5–15 frames · same group · JPEG / PNG</p>
            </div>
          )}

          {fileError && (
            <p className="mt-2 font-mono text-[11px] text-[#B23A2E] flex items-center gap-1.5">
              <AlertTriangle className="w-3 h-3" /> {fileError}
            </p>
          )}

          {/* Uploading spinner */}
          {stage === STAGE.UPLOADING && (
            <div className="flex items-center gap-3 mt-4">
              <Loader2 className="w-5 h-5 text-[#2F7A78] animate-spin" />
              <span className="font-mono text-[12px] text-[#6b6354]">Uploading {filePreviews.length} frames…</span>
            </div>
          )}
        </div>
      </section>

      {/* Filmstrip preview strip (shown once files are selected) */}
      {filePreviews.length > 0 && (
        <div id="s-roll" className="mb-12">
          <span className="font-hand text-[19px] text-[#1A1610] -rotate-[1deg] inline-block mb-3">the roll, as shot</span>
          <div className="filmstrip relative bg-[#1B1611] py-[18px] px-[4px] flex shadow-[0_16px_32px_rgba(0,0,0,0.25)] overflow-x-auto">
            {filePreviews.map((src, i) => (
              <div key={i} className="flex-shrink-0 relative border-r border-white/5 last:border-r-0 px-1">
                <span className="font-mono text-[8px] text-[rgba(228,233,230,0.4)] absolute top-1 left-2 tracking-widest">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <div className="w-[90px] h-[68px] mt-4 overflow-hidden">
                  <img src={src} alt={`frame ${i + 1}`} className="w-full h-full object-cover" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 2 — Pipeline tracker + Receipt (visible while processing / complete)
      ══════════════════════════════════════════════════════════════════════ */}
      {stage !== STAGE.IDLE && (
        <section id="s-pipeline" className="mb-12 flex gap-8 flex-wrap items-start">

          {/* Stage tracker */}
          <div className="flex-1 min-w-[260px]">
            <span className="font-hand text-[19px] text-[#1A1610] -rotate-[1deg] inline-block mb-4">verification pipeline</span>
            <div className="flex flex-col gap-3">
              {PIPELINE_STEPS.map(step => {
                const s = stepStatus(step.id, stage, stage);
                return (
                  <div
                    key={step.id}
                    className={`flex items-start gap-3 p-3 transition-all ${
                      s === 'active' ? 'bg-[#F7F1E1] shadow-sm' :
                      s === 'done'   ? 'opacity-80' :
                      s === 'error'  ? 'bg-red-50' : 'opacity-40'
                    }`}
                  >
                    <StepIcon status={s} />
                    <div>
                      <p className={`text-[13px] font-semibold ${s === 'active' ? 'text-[#1A1610]' : 'text-[#6b6354]'}`}>
                        {step.label}
                      </p>
                      <p className="text-[11px] font-mono text-[#6b6354]/70 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Verification receipt */}
          <div
            className="bg-[#F7F1E1] w-[min(360px,100%)] px-[22px] py-[20px] pb-[32px] rotate-[0.6deg] shadow-[0_14px_26px_rgba(20,15,10,0.18)]"
            style={{ clipPath: 'polygon(0 0,100% 0,100% 93%,95% 100%,89% 91%,83% 100%,77% 91%,71% 100%,65% 91%,59% 100%,53% 91%,47% 100%,41% 91%,35% 100%,29% 91%,23% 100%,17% 91%,11% 100%,5% 91%,0 100%)' }}
          >
            <p className="font-mono text-[10px] tracking-widest text-[#6b6354] mb-3 border-b border-dashed border-black/10 pb-2">
              VERIFICATION · PRINTOUT
            </p>

            {/* Progress bar */}
            <div className="w-full h-1.5 bg-black/8 rounded overflow-hidden mb-4">
              <div
                className="h-full bg-[#2F7A78] transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="flex flex-col gap-0 font-mono text-[12px]">
              {[
                ['GATE 1 — INPUT',
                  progress > 15
                    ? (result?.gate_1_result?.passed !== false ? 'valid ✓' : 'rejected ✗')
                    : stage === STAGE.GATE1 ? 'checking…' : 'standing by'],
                ['GATE 2 — OUTPUT',
                  isComplete
                    ? (result?.gate_2_result?.passed ? `natural · ${result.gate_2_result.confidence?.toFixed(2)} ✓` : 'flagged ✗')
                    : [STAGE.GATE2, STAGE.RETRYING].includes(stage) ? 'inspecting…' : 'standing by'],
                ['RETRY ATTEMPTS',
                  isComplete ? `${result?.retry_count ?? 0} / 2` : '—'],
                ['RESULT',
                  isComplete
                    ? (result?.result_type === 'blended' ? 'blended composite' : 'best single frame')
                    : '—'],
              ].map(([label, val]) => (
                <div key={label} className="flex justify-between gap-3 py-[5px] border-b border-dashed border-black/10 last:border-b-0">
                  <span className="text-[#6b6354]">{label}</span>
                  <span className={
                    val.includes('✓') ? 'text-[#4B7A51] font-bold' :
                    val.includes('✗') ? 'text-[#B23A2E] font-bold' :
                    val.includes('…') ? 'text-[#C89B3C] animate-pulse' :
                    'text-[#6b6354]'
                  }>{val}</span>
                </div>
              ))}
            </div>

            {/* Live message */}
            {message && (
              <p className="font-mono text-[10px] text-[#6b6354]/60 mt-3 border-t border-dashed border-black/10 pt-2 truncate">
                ↳ {message}
              </p>
            )}
          </div>
        </section>
      )}

      {/* ── Error cards ── */}
      {stage === STAGE.ERROR && (
        <ErrorCard
          error={error}
          onRetry={burstId ? retry : null}
          onReset={reset}
        />
      )}

      {/* Gate 2 soft warning (result still shown, but flagged) */}
      {isComplete && error?.type === 'gate2' && (
        <div className="mb-8 p-4 bg-[#fff8f0] border border-[#C89B3C]/50 shadow flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-[#C89B3C] flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-mono text-[12px] font-bold text-[#1A1610] mb-1">Gate 2 flagged blending issues</p>
            {error.issues.map((iss, i) => (
              <p key={i} className="font-mono text-[11px] text-[#6b6354]">
                [{iss.person_cluster_id || 'global'}] {iss.issue_type} — {iss.description}
              </p>
            ))}
            <button
              onClick={retry}
              className="mt-2 font-mono text-[11px] flex items-center gap-1.5 bg-[#C89B3C] hover:bg-[#b58932] text-white px-3 py-1 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#C89B3C]"
            >
              <RefreshCw className="w-3 h-3" /> Re-trigger with next-best faces
            </button>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 3 — Results: picks + compare slider + keeper Polaroid
      ══════════════════════════════════════════════════════════════════════ */}
      {isComplete && result && (
        <section id="s-result">

          {/* Face selection chips */}
          {result.per_person_reasoning?.length > 0 && (
            <div className="mb-10">
              <span className="font-hand text-[19px] text-[#1A1610] -rotate-[1deg] inline-block mb-4">who was picked, and why</span>
              <div className="flex flex-wrap gap-2">
                {result.per_person_reasoning.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 bg-[#F7F1E1] px-3 py-2 text-[12px] shadow-[0_3px_8px_rgba(20,15,10,0.12)]">
                    <span className="w-2 h-2 rounded-full bg-[#2F7A78] flex-shrink-0" />
                    <span className="font-mono font-bold text-[#1A1610]">{r.cluster_id}</span>
                    <span className="font-mono text-[#6b6354]">· frame {r.selected_frame}</span>
                    {r.reason && (
                      <span className="font-mono text-[10px] text-[#6b6354]/60 hidden sm:inline">— {r.reason}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Before / After comparison slider */}
          <div className="mb-12">
            <span className="font-hand text-[19px] text-[#1A1610] -rotate-[1deg] inline-block mb-3">drag to compare</span>
            <p className="font-mono text-[12px] text-[#6b6354] mb-4">
              Drag the divider left/right. Left = original base frame · Right = Kairos composite.
            </p>
            <CompareSlider
              before={filePreviews[0] || result.output_image_url}
              after={result.output_image_url}
            />
            <p className="font-mono text-[10px] text-[#6b6354]/50 mt-2 text-center">← drag to reveal →</p>
          </div>

          {/* Final keeper Polaroid */}
          <div className="flex justify-center mb-16">
            <div className="relative">
              {/* Rejected frames blurred behind */}
              <div className="absolute inset-0 flex items-center justify-center gap-3 blur-[4px] saturate-50 opacity-20 pointer-events-none">
                {filePreviews.slice(0, 3).map((src, i) => (
                  <div key={i} className="w-[70px] h-[70px] overflow-hidden" style={{ transform: `rotate(${[-8,5,-3][i]}deg)` }}>
                    <img src={src} alt="" className="w-full h-full object-cover" />
                  </div>
                ))}
              </div>

              {/* The keeper */}
              <div className="relative z-10 bg-[#F7F1E1] p-[16px_16px_32px] w-[340px] shadow-[0_26px_50px_rgba(20,15,10,0.20)]">
                <div className="aspect-[4/3] overflow-hidden bg-[#1b1712]">
                  <img src={result.output_image_url} alt="final composite" className="w-full h-full object-cover" />
                </div>

                <p className="font-mono text-[11px] text-[#6b6354] text-center mt-3">
                  {result.result_type === 'blended'
                    ? <>composite · gate 2 confidence <b className="text-[#B23A2E]">{result.gate_2_result?.confidence?.toFixed(2) ?? '—'}</b></>
                    : <><Sparkles className="inline w-3 h-3 mr-1 text-[#C89B3C]" />best unedited frame · safe fallback</>
                  }
                </p>

                <div className="flex gap-2 mt-4">
                  <a
                    href={result.output_image_url}
                    download={`kairos_${burstId?.slice(0, 6) ?? 'result'}.jpg`}
                    className="flex-1 flex items-center justify-center gap-2 bg-[#1A1610] hover:bg-[#2A2521] text-[#F7F1E1] font-mono text-[11px] tracking-wide py-2.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1A1610]"
                  >
                    <Download className="w-3.5 h-3.5" /> DOWNLOAD KEEPER
                  </a>
                  <button
                    onClick={reset}
                    className="px-3 bg-[#1A1610]/8 hover:bg-[#1A1610]/15 text-[#6b6354] font-mono text-[11px] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1A1610]"
                    title="Process a new burst"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      <footer className="font-mono text-center pt-8 pb-2 text-[10px] text-[#6b6354]/60 border-t border-[#1A1610]/8">
        KAIROS LIGHT TABLE v3 · PERFECT GROUP PHOTO BLENDER · VIT-AP HACKATHON 2026
      </footer>
    </div>
  );
}
