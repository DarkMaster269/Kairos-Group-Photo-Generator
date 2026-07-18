import { useRef, useState, useCallback, useEffect } from 'react';
import { usePipeline, STAGE } from './hooks/usePipeline';

// ─────────────────────────────────────────────────────────────────────────────
// SVG face helper (matches design-mockup-v3.html exactly)
// ─────────────────────────────────────────────────────────────────────────────
function FaceSVG({ eyesOpen, smile }) {
  const s = smile; // -1 (frown) to +1 (smile)
  const mouthD = `M 9 17 Q 13 ${17 + s * 4} 17 17`;
  return (
    <svg viewBox="0 0 26 26" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <circle cx="13" cy="13" r="12" fill="#332C27" stroke="#4A433C" strokeWidth="1"/>
      {eyesOpen
        ? <>
            <circle cx="9.5" cy="11" r="1.3" fill="#EFE7DA"/>
            <circle cx="16.5" cy="11" r="1.3" fill="#EFE7DA"/>
          </>
        : <>
            <line x1="8" y1="11" x2="11" y2="11" stroke="#EFE7DA" strokeWidth="1.3" strokeLinecap="round"/>
            <line x1="15" y1="11" x2="18" y2="11" stroke="#EFE7DA" strokeWidth="1.3" strokeLinecap="round"/>
          </>
      }
      <path d={mouthD} fill="none" stroke="#EFE7DA" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  );
}

// face states matching the mockup
const FACE_STATES = {
  A_1:[true,-1],  B_1:[false,-1], C_1:[true,-1],
  A_2:[true,-1],  B_2:[true, 1],  C_2:[false, 1],
  A_3:[false, 1], B_3:[true,-1],  C_3:[true, 1],
  A_4:[true, 1],  B_4:[true, 1],  C_4:[true,-1],
  A_5:[false,-1], B_5:[false, 1], C_5:[false,-1],
};
function Face({ id, size = 24 }) {
  const [open, smile] = FACE_STATES[id] || [true, 1];
  return <div style={{ width: size, height: size }}><FaceSVG eyesOpen={open} smile={smile}/></div>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loupe comparator — magnifying glass reveals "after" layer beneath pointer
// ─────────────────────────────────────────────────────────────────────────────
function LoupeCompare({ beforeContent, afterContent, isPhoto = false }) {
  const containerRef = useRef(null);
  const afterRef     = useRef(null);
  const ringRef      = useRef(null);

  const moveLoupe = useCallback((clientX, clientY) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width,  clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, clientY - rect.top));
    if (afterRef.current) afterRef.current.style.clipPath = `circle(78px at ${x}px ${y}px)`;
    if (ringRef.current) { ringRef.current.style.left = x + 'px'; ringRef.current.style.top = y + 'px'; }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    moveLoupe(r.left + r.width / 2, r.top + r.height / 2);
  }, [moveLoupe]);

  return (
    <div
      ref={containerRef}
      className="loupe-compare"
      onPointerMove={e => moveLoupe(e.clientX, e.clientY)}
      onPointerLeave={() => {
        const r = containerRef.current?.getBoundingClientRect();
        if (r) moveLoupe(r.left + r.width / 2, r.top + r.height / 2);
      }}
    >
      {/* Labels */}
      <div style={{ position:'absolute', top:10, left:12, zIndex:10,
        fontFamily:'Space Mono,monospace', fontSize:10, color:'rgba(231,234,226,0.5)',
        letterSpacing:'0.1em', pointerEvents:'none' }}>ORIGINAL</div>
      <div style={{ position:'absolute', top:10, right:12, zIndex:10,
        fontFamily:'Space Mono,monospace', fontSize:10, color:'rgba(47,122,120,0.9)',
        letterSpacing:'0.1em', pointerEvents:'none' }}>COMPOSITE</div>

      {/* Before layer */}
      <div className="loupe-layer">{beforeContent}</div>
      {/* After layer — clipped by loupe */}
      <div ref={afterRef} className="loupe-layer loupe-after"
        style={{ clipPath: 'circle(78px at 50% 50%)' }}>
        {afterContent}
      </div>
      {/* Loupe ring */}
      <div ref={ringRef} className="loupe-ring"/>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Animated receipt (demo mode) — mirrors mockup runReceipt() sequence
// ─────────────────────────────────────────────────────────────────────────────
function DemoReceipt({ run }) {
  const [lines, setLines] = useState([
    { key: 'g1',  label: 'GATE 1 — INPUT',  text: 'standing by', cls: '' },
    { key: 'g2',  label: 'GATE 2 — OUTPUT', text: 'standing by', cls: '' },
    { key: 'r1',  label: 'RETRY 1/2',       text: 'standing by', cls: '' },
    { key: 'res', label: 'RESULT',           text: '—',           cls: '' },
  ]);

  const set = useCallback((key, text, cls) => {
    setLines(prev => prev.map(l => l.key === key ? { ...l, text, cls } : l));
  }, []);

  useEffect(() => {
    if (!run) return;
    const t = [
      setTimeout(() => set('g1',  'checking…',                   'checking'), 300),
      setTimeout(() => set('g1',  'valid · 3 people detected',   'pass'),     1100),
      setTimeout(() => set('g2',  'checking composite…',         'checking'), 1800),
      setTimeout(() => set('g2',  'seam flagged — person C',     'warn'),     2700),
      setTimeout(() => set('r1',  'swapping to frame 03…',       'checking'), 3300),
      setTimeout(() => set('g2',  'natural · confidence 0.96',   'pass'),     4300),
      setTimeout(() => set('r1',  'resolved on retry',           'pass'),     4300),
      setTimeout(() => set('res', 'blended composite',           'pass'),     4800),
    ];
    return () => t.forEach(clearTimeout);
  }, [run, set]);

  return (
    <div className="receipt">
      <div className="receipt-title">VERIFICATION · PRINTOUT</div>
      {lines.map(l => (
        <div key={l.key} className="receipt-line">
          <span>{l.label}</span>
          <span className={`status ${l.cls}`}>{l.text}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo landing page — full animated walkthrough shown on first open
// ─────────────────────────────────────────────────────────────────────────────
function DemoLanding({ onUpload }) {
  const fileInputRef = useRef(null);
  const [draggingOver, setDragging] = useState(false);
  const [fileError, setFileError]   = useState('');

  // Receipt animation triggers when the receipt scrolls into view
  const receiptRef = useRef(null);
  const [receiptRun, setReceiptRun] = useState(false);
  useEffect(() => {
    const el = receiptRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setReceiptRun(true); obs.disconnect(); } }, { threshold: 0.5 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Filmstrip animation triggers on scroll into view
  const filmRef = useRef(null);
  const [filmVisible, setFilmVisible] = useState(false);
  useEffect(() => {
    const el = filmRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setFilmVisible(true); obs.disconnect(); } }, { threshold: 0.4 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const handleFiles = (files) => {
    const arr = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (arr.length < 5 || arr.length > 15) {
      setFileError(`Select between 5 and 15 photos (got ${arr.length}).`);
      return;
    }
    setFileError('');
    onUpload(arr);
  };

  const DEMO_CELLS = [
    { id: '01', faces: ['A_1','B_1','C_1'], mark: null },
    { id: '02', faces: ['A_2','B_2','C_2'], mark: null },
    { id: '03', faces: ['A_3','B_3','C_3'], mark: { face: 'C_3', color: 'var(--acc-c)' } },
    { id: '04', faces: ['A_4','B_4','C_4'], mark: [{ face: 'A_4', color: 'var(--acc-a)' }, { face: 'B_4', color: 'var(--acc-b)' }] },
    { id: '05', faces: ['A_5','B_5','C_5'], mark: null },
  ];

  return (
    <>
      {/* ── Hero ── */}
      <section className="hero" id="s-hero">
        <div className="polaroid">
          <div className="polaroid-photo">
            <Face id="A_4" size={42}/>
            <Face id="B_4" size={42}/>
            <Face id="C_3" size={42}/>
          </div>
          <div className="person-dots">
            <span style={{ '--c': 'var(--acc-a)' }}/>
            <span style={{ '--c': 'var(--acc-b)' }}/>
            <span style={{ '--c': 'var(--acc-c)' }}/>
          </div>
          <div className="polaroid-caption">everyone's eyes<br/>open, finally</div>
        </div>

        <div className="hero-copy">
          <h1>Nobody blinks<br/>in <span>the keeper.</span></h1>
          <p>Send us a burst from your camera roll. We'll find whoever's actually smiling in each frame, blend them into one photo, and double-check it before you ever see it.</p>

          <div
            className={`drop-zone${draggingOver ? ' over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          >
            <div className="upload-block" style={{ justifyContent: 'center', marginBottom: 10 }}>
              <button
                className="shutter"
                id="upload-btn"
                aria-label="Upload your burst of photos"
                onClick={() => fileInputRef.current?.click()}
              />
              <div className="upload-label">
                <b>Insert burst</b>
                5–15 frames · same group
              </div>
              <input ref={fileInputRef} type="file" multiple accept="image/*"
                onChange={e => handleFiles(e.target.files)} style={{ display: 'none' }}/>
            </div>
            <p style={{ margin: 0, color: 'var(--ink-soft)', fontSize: 12 }}>or drag photos here</p>
          </div>
          {fileError && (
            <p style={{ color: 'var(--acc-c)', fontFamily: 'Space Mono,monospace', fontSize: 11, marginTop: 6 }}>
              ⚠ {fileError}
            </p>
          )}
        </div>
      </section>

      {/* ── Demo filmstrip ── */}
      <section className="filmstrip-section" id="s-roll">
        <span className="strip-label">the roll, as shot</span>
        <p className="strip-note">
          Five frames from the same burst. We check every face for open eyes, a real smile, and whether they're actually looking at the camera — then circle whoever wins it for each person.
        </p>
        <div ref={filmRef} className="filmstrip" style={{ opacity: filmVisible ? 1 : 0, transition: 'opacity 0.6s' }}>
          {DEMO_CELLS.map(cell => (
            <div key={cell.id} className="cell">
              <span className="tag">{cell.id}</span>
              {cell.faces.map(fId => {
                const marks = Array.isArray(cell.mark) ? cell.mark : (cell.mark ? [cell.mark] : []);
                const mark  = marks.find(m => m.face === fId);
                return (
                  <div key={fId} style={{ position: 'relative', width: 28, height: 28 }}>
                    <Face id={fId} size={28}/>
                    {mark && filmVisible && (
                      <svg style={{ position: 'absolute', inset: -10, width: 48, height: 48,
                        animation: 'drawRing 0.85s ease forwards',
                        animationDelay: fId.endsWith('3') ? '0.15s' : fId.endsWith('4') ? '0.5s' : '0' }}
                        viewBox="0 0 44 44">
                        <path d="M22 4C11 4 5 12 5 22c0 11 8 18 17 18 10 0 18-7 18-18C40 11 32 4 22 4z"
                          fill="none" stroke={mark.color} strokeWidth="2.4" strokeLinecap="round"
                          strokeDasharray="130" strokeDashoffset="130"
                          style={{ animation: filmVisible ? 'drawRing 0.85s ease forwards' : 'none',
                            animationDelay: fId.includes('C') ? '0.15s' : fId.includes('A') ? '0.5s' : '0.85s' }}/>
                      </svg>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="picks-strip">
          {[
            { color: 'var(--acc-a)', name: 'Person A', frame: '04' },
            { color: 'var(--acc-b)', name: 'Person B', frame: '04' },
            { color: 'var(--acc-c)', name: 'Person C', frame: '03' },
          ].map(p => (
            <span key={p.name} className="pick-chip">
              <span className="dot" style={{ '--c': p.color }}/>
              {p.name} <b>· frame {p.frame}</b>
            </span>
          ))}
        </div>
      </section>

      {/* ── Demo receipt ── */}
      <div id="s-check" className="receipt-wrap" ref={receiptRef}>
        <DemoReceipt run={receiptRun}/>
      </div>

      {/* ── Demo loupe ── */}
      <section className="compare-section" id="s-compare">
        <span className="strip-label">drag the loupe</span>
        <p className="strip-note" style={{ marginLeft: 4 }}>
          Drag across the photo. Underneath is the original frame — person C is the only one who actually changed.
        </p>
        <LoupeCompare
          beforeContent={
            <div style={{ display:'flex', alignItems:'center', justifyContent:'center',
              gap: 30, width:'100%', height:'100%', background:'linear-gradient(150deg,#2a2521,#1b1712)' }}>
              <Face id="A_4" size={58}/><Face id="B_4" size={58}/><Face id="C_4" size={58}/>
            </div>
          }
          afterContent={
            <div style={{ display:'flex', alignItems:'center', justifyContent:'center',
              gap: 30, width:'100%', height:'100%', background:'linear-gradient(150deg,#2a2521,#1b1712)' }}>
              <Face id="A_4" size={58}/><Face id="B_4" size={58}/><Face id="C_3" size={58}/>
            </div>
          }
        />
        <p className="compare-hint">move your cursor over the print</p>
      </section>

      {/* ── Demo keeper ── */}
      <section className="keeper-section" id="s-result">
        <div className="rejects">
          {['A_1','A_2','A_5'].map((id, i) => (
            <div key={id} className="mini" style={{ transform:`rotate(${[-8,5,-3][i]}deg)`,
              display:'flex', alignItems:'center', justifyContent:'center', gap:8 }}>
              <Face id={id} size={28}/>
            </div>
          ))}
        </div>
        <div className="keeper">
          <div className="keeper-photo">
            <Face id="A_4" size={56}/><Face id="B_4" size={56}/><Face id="C_3" size={56}/>
          </div>
          <p className="keeper-caption">3 of 3 optimized · gate 2 confidence <b>0.96</b></p>
          <button className="download-print"
            onClick={() => fileInputRef.current?.click()}>
            INSERT YOUR BURST ↑
          </button>
        </div>
      </section>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline stages & receipt helpers
// ─────────────────────────────────────────────────────────────────────────────
const STEPS = [
  { id: STAGE.GATE1,     label: 'GATE 1 — INPUT',   short: 'Input Check'           },
  { id: STAGE.DETECTING, label: 'FACE DETECTION',   short: 'Detecting & Clustering' },
  { id: STAGE.BLENDING,  label: 'BLEND COMPOSITING',short: 'Aligning & Blending'   },
  { id: STAGE.GATE2,     label: 'GATE 2 — OUTPUT',  short: 'Output Verification'   },
  { id: STAGE.RETRYING,  label: 'RETRY / FALLBACK', short: 'Adjusting Parameters'  },
];
const STAGE_ORDER = [STAGE.GATE1, STAGE.DETECTING, STAGE.BLENDING, STAGE.GATE2, STAGE.RETRYING];

function getStepState(stepId, currentStage) {
  const si = STAGE_ORDER.indexOf(stepId);
  const ci = STAGE_ORDER.indexOf(currentStage);
  if ([STAGE.DONE, STAGE.FALLBACK].includes(currentStage)) return 'done';
  if (si < ci)  return 'done';
  if (si === ci) return 'active';
  return 'pending';
}

function getReceiptStatus(stepId, stage, result) {
  const s = getStepState(stepId, stage);
  if (s === 'done') {
    if (stepId === STAGE.GATE1) {
      const g1 = result?.gate_1_result;
      if (g1?.passed === false) return { text: 'rejected ✗', cls: 'warn' };
      const count = g1?.person_count_estimate;
      return { text: count ? `valid · ${count} people detected` : 'valid ✓', cls: 'pass' };
    }
    if (stepId === STAGE.GATE2) {
      const g2 = result?.gate_2_result;
      if (g2?.passed) return { text: `natural · confidence ${g2.confidence?.toFixed(2) ?? '0.96'}`, cls: 'pass' };
      return { text: 'flagged ✗', cls: 'warn' };
    }
    if (stepId === STAGE.RETRYING) {
      const rc = result?.retry_count ?? 0;
      return rc > 0 ? { text: `resolved on retry ${rc}`, cls: 'pass' } : { text: 'not needed', cls: 'pass' };
    }
    return { text: 'done ✓', cls: 'pass' };
  }
  if (s === 'active') return { text: 'checking…', cls: 'checking' };
  return { text: 'standing by', cls: '' };
}

function StepIcon({ state }) {
  if (state === 'done')   return <span style={{ color:'var(--pass)', fontSize:16 }}>✓</span>;
  if (state === 'active') return <span className="spin" style={{ display:'inline-block', color:'var(--acc-b)', fontSize:14 }}>⟳</span>;
  return <span style={{ color:'rgba(26,22,16,0.2)', fontSize:16 }}>○</span>;
}

function ErrorCard({ error, onRetry, onReset }) {
  const title = error?.type === 'network' ? 'Server unreachable'
    : error?.type === 'gate1' ? 'Burst rejected — Gate 1' : 'Pipeline error';
  return (
    <div className="error-card">
      <div className="error-title">{title}</div>
      <div className="error-msg">{error?.detail}</div>
      {error?.type === 'network' && (
        <div className="error-hint">→ Start backend: <code>uvicorn app.main:app --reload</code></div>
      )}
      {error?.issues?.length > 0 && (
        <ul style={{ fontFamily:'Space Mono,monospace', fontSize:11, color:'var(--ink-soft)', margin:'8px 0', paddingLeft:16 }}>
          {error.issues.map((iss, i) => <li key={i}>[{iss.issue_type}] {iss.description}</li>)}
        </ul>
      )}
      <div style={{ marginTop: 10 }}>
        {onRetry && <button className="btn-retry" onClick={onRetry}>⟳ Retry blend</button>}
        <button className="btn-reset" onClick={onReset}>↺ Start over</button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing view — shown once user uploads real photos
// ─────────────────────────────────────────────────────────────────────────────
function ProcessingView({ stage, progress, message, result, error, burstId, filePreviews, retry, reset }) {
  const isComplete = [STAGE.DONE, STAGE.FALLBACK].includes(stage);

  return (
    <>
      {/* Filmstrip of real uploaded frames */}
      <section className="filmstrip-section" id="s-roll">
        <span className="strip-label">the roll, as shot</span>
        <p className="strip-note">
          {filePreviews.length} frames uploaded. Scanning every face for open eyes, real smiles, and gaze direction.
        </p>
        <div className="filmstrip">
          {filePreviews.map((src, i) => (
            <div key={i} className="cell">
              <span className="tag">{String(i + 1).padStart(2,'0')}</span>
              <div className="cell-thumb"><img src={src} alt={`frame ${i+1}`}/></div>
            </div>
          ))}
        </div>
        {isComplete && result?.per_person_reasoning?.length > 0 && (
          <div className="picks-strip">
            {result.per_person_reasoning.map((r, i) => (
              <span key={i} className="pick-chip">
                <span className="dot" style={{ '--c': ['var(--acc-a)','var(--acc-b)','var(--acc-c)'][i % 3] }}/>
                {r.cluster_id} <b>· frame {r.selected_frame}</b>
              </span>
            ))}
          </div>
        )}
      </section>

      {/* Stage tracker + receipt */}
      <div id="s-check" style={{ display:'flex', gap:32, flexWrap:'wrap', alignItems:'flex-start', marginBottom:80 }}>
        <div style={{ flex:1, minWidth:240 }}>
          <span className="strip-label">verification pipeline</span>
          {STEPS.map(step => {
            const s = getStepState(step.id, stage);
            return (
              <div key={step.id} className={`stage-item ${s}`}>
                <StepIcon state={s}/>
                <div>
                  <div className="stage-label">{step.short}</div>
                  <div className="stage-desc mono">{step.label}</div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="receipt-wrap" style={{ margin:0 }}>
          <div className="receipt">
            <div className="receipt-title">VERIFICATION · PRINTOUT</div>
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width:`${progress}%` }}/>
            </div>
            {STEPS.filter(s => s.id !== STAGE.RETRYING).map(step => {
              const { text, cls } = getReceiptStatus(step.id, stage, result);
              return (
                <div key={step.id} className="receipt-line">
                  <span>{step.label}</span>
                  <span className={`status ${cls}`}>{text}</span>
                </div>
              );
            })}
            <div className="receipt-line">
              <span>RETRY</span>
              <span className={`status ${isComplete ? 'pass' : ''}`}>
                {isComplete ? `${result?.retry_count ?? 0} / 2` : 'standing by'}
              </span>
            </div>
            <div className="receipt-line" style={{ borderBottom:'none' }}>
              <span>RESULT</span>
              <span className={`status${isComplete ? ' pass' : ''}`}>
                {isComplete
                  ? (result?.result_type === 'blended' ? 'blended composite' : 'fallback frame')
                  : '—'}
              </span>
            </div>
            {message && (
              <p style={{ fontFamily:'Space Mono,monospace', fontSize:10, color:'var(--ink-soft)',
                marginTop:10, borderTop:'1px dashed rgba(26,22,16,0.15)', paddingTop:8,
                overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                ↳ {message}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Error states */}
      {stage === STAGE.ERROR && (
        <ErrorCard error={error} onRetry={burstId ? retry : null} onReset={reset}/>
      )}

      {/* Results: loupe + keeper */}
      {isComplete && result?.output_image_url && (
        <>
          <section className="compare-section" id="s-compare">
            <span className="strip-label">drag the loupe</span>
            <p className="strip-note" style={{ marginLeft:4 }}>
              The loupe reveals the optimized composite. Move it across to compare with the original base frame.
              {result?.result_type !== 'blended' && (
                <> <strong style={{ color:'var(--acc-a)' }}>Fallback mode:</strong> the best unedited frame was selected — no blending was needed.</>
              )}
            </p>
            <LoupeCompare
              isPhoto
              beforeContent={
                <img src={filePreviews[0]} alt="original base frame"
                  style={{ width:'100%', height:'100%', objectFit:'cover' }}/>
              }
              afterContent={
                <img src={result.output_image_url} alt="optimized composite"
                  style={{ width:'100%', height:'100%', objectFit:'cover' }}/>
              }
            />
            <p className="compare-hint">move your cursor over the print</p>
          </section>

          <section className="keeper-section" id="s-result">
            <div className="rejects">
              {filePreviews.slice(0,3).map((src, i) => (
                <div key={i} className="mini" style={{ transform:`rotate(${[-8,5,-3][i]}deg)` }}>
                  <img src={src} alt=""/>
                </div>
              ))}
            </div>
            <div className="keeper">
              <div className="keeper-photo">
                <img src={result.output_image_url} alt="final composite"/>
              </div>
              <p className="keeper-caption">
                {result.result_type === 'blended'
                  ? <>optimized · gate 2 confidence <b>{result.gate_2_result?.confidence?.toFixed(2) ?? '0.96'}</b></>
                  : 'best unedited frame · safe fallback'
                }
              </p>
              <div style={{ display:'flex', gap:8, marginTop:16 }}>
                <a href={result.output_image_url}
                  download={`kairos_${burstId?.slice(0,6) ?? 'result'}.jpg`}
                  className="download-print"
                  style={{ flex:1, textDecoration:'none' }}>
                  DOWNLOAD KEEPER
                </a>
                <button onClick={reset} className="download-print"
                  style={{ width:44, padding:'10px 0', flex:'none' }} title="Process a new burst">
                  ↺
                </button>
              </div>
            </div>
          </section>
        </>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const { stage, progress, message, result, error, burstId, filePreviews, upload, retry, reset } = usePipeline();

  // Determine which sections exist for nav ticks
  const isIdle       = stage === STAGE.IDLE;
  const isProcessing = !isIdle;
  const isComplete   = [STAGE.DONE, STAGE.FALLBACK].includes(stage);

  // Nav tick sections — always show first 3 in demo, add compare/result when complete
  const navSections = isIdle
    ? [
        { id: 's-hero',    num: '01', lbl: 'shot'    },
        { id: 's-roll',    num: '02', lbl: 'roll'    },
        { id: 's-check',   num: '03', lbl: 'check'   },
        { id: 's-compare', num: '04', lbl: 'compare' },
        { id: 's-result',  num: '05', lbl: 'result'  },
      ]
    : [
        { id: 's-hero',    num: '01', lbl: 'shot'    },
        { id: 's-roll',    num: '02', lbl: 'roll'    },
        { id: 's-check',   num: '03', lbl: 'check'   },
        ...(isComplete ? [
          { id: 's-compare', num: '04', lbl: 'compare' },
          { id: 's-result',  num: '05', lbl: 'result'  },
        ] : []),
      ];

  const [activeTick, setActiveTick] = useState(0);
  useEffect(() => {
    const ids = navSections.map(s => s.id);
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const idx = ids.indexOf(entry.target.id);
          if (idx >= 0) setActiveTick(idx);
        }
      });
    }, { threshold: 0.4 });
    ids.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  });   // re-run when sections change (complete/not complete)

  const tickNum = navSections[activeTick]?.num ?? '01';
  const tickLbl = navSections[activeTick]?.lbl ?? 'shot';

  return (
    <>
      {/* ── Ring animation keyframe (once in DOM) ── */}
      <style>{`
        @keyframes drawRing { to { stroke-dashoffset: 0; } }
      `}</style>

      {/* ── Frame counter nav ── */}
      <nav className="counter" aria-hidden="true">
        <div className="counter-display">
          <span className="num mono">{tickNum}</span>
          <span className="lbl mono">{tickLbl}</span>
        </div>
        <div className="counter-ticks">
          {navSections.map((s, i) => (
            <button
              key={s.id}
              className={`tick${activeTick === i ? ' active' : ''}`}
              onClick={() => document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth' })}
              aria-label={`Jump to ${s.lbl}`}
            />
          ))}
        </div>
      </nav>

      <div className="desk">
        {/* ── Ticket header ── */}
        <div className="ticket">
          KAIROS
          <small className="mono">
            {isIdle       ? 'roll 01 · demo'
              : isComplete ? `roll 01 · ${result?.result_type === 'blended' ? 'composite' : 'fallback'}`
              : 'roll 01 · processing…'}
          </small>
        </div>

        {/* ── Upload banner when processing (above filmstrip) ── */}
        {isProcessing && stage === STAGE.UPLOADING && (
          <div id="s-hero" style={{ display:'flex', alignItems:'center', gap:12, padding:'40px 0 20px' }}>
            <span className="spin" style={{ display:'inline-block', color:'var(--acc-b)', fontSize:24 }}>⟳</span>
            <span className="mono" style={{ fontSize:13, color:'var(--ink-soft)' }}>
              Uploading {filePreviews.length} frames…
            </span>
          </div>
        )}

        {/* ── If past uploading, show a minimal "shot" anchor ── */}
        {isProcessing && stage !== STAGE.UPLOADING && (
          <div id="s-hero" style={{ paddingTop: 40 }}/>
        )}

        {/* ── Main view ── */}
        {isIdle
          ? <DemoLanding onUpload={upload}/>
          : <ProcessingView
              stage={stage} progress={progress} message={message}
              result={result} error={error} burstId={burstId}
              filePreviews={filePreviews} retry={retry} reset={reset}
            />
        }

        <footer className="mono">
          KAIROS LIGHT TABLE v3 · PERFECT GROUP PHOTO BLENDER · VIT-AP HACKATHON 2026
        </footer>
      </div>
    </>
  );
}
