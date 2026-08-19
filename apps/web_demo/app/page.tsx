"use client";

/* Uploaded blob previews and API-returned data URLs cannot use Next image optimization. */
/* eslint-disable @next/next/no-img-element */

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type ModelKey = "unet" | "segformer" | "vmamba";
type ModelStatus = { available: boolean; checkpoint: string; reason?: string; policy_compatible?: boolean };
type PolicyStatus = { available: boolean; ready?: boolean; path: string; models: string[]; mode?: string; targets?: Record<string, number> };
type Result = {
  model: string;
  threshold: number;
  min_component_area_px: number;
  state: "pass" | "review" | "strong";
  strong_component_count: number;
  review_component_count: number;
  elapsed_seconds: number;
  mask_png: string;
  overlay_png: string;
};
type Decision = {
  level: "pass" | "review" | "defect";
  reason: string;
  required_votes: number;
  max_spatial_votes: number;
  strong_models: string[];
  candidate_models: string[];
  mask_pixels: number;
  mask_png: string;
  overlay_png: string;
};

const models: { key: ModelKey; name: string; subtitle: string }[] = [
  { key: "unet", name: "U-Net / ResNet-18", subtitle: "CNN baseline" },
  { key: "segformer", name: "SegFormer-B0", subtitle: "Transformer encoder" },
  { key: "vmamba", name: "VMamba-T", subtitle: "State-space encoder" },
];

const apiBase = process.env.NEXT_PUBLIC_INFERENCE_API ?? "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [selected, setSelected] = useState<ModelKey[]>([]);
  const [status, setStatus] = useState<Record<string, ModelStatus>>({});
  const [policy, setPolicy] = useState<PolicyStatus | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [message, setMessage] = useState("Choose an image to begin.");
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Inference service health check failed.");
        return response.json();
      })
      .then((data) => {
        const nextStatus = data.models ?? {};
        const readyModels = models
          .filter(({ key }) => nextStatus[key]?.available && nextStatus[key]?.policy_compatible !== false)
          .map(({ key }) => key);
        setStatus(nextStatus);
        setPolicy(data.policy ?? null);
        setSelected(readyModels);
        if (!readyModels.length) {
          setMessage("No trained checkpoints are available. Follow the inference setup first.");
        }
      })
      .catch(() => setMessage("Inference service is offline. Start the local Python service first."));
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const availableCount = useMemo(
    () => selected.filter((key) => status[key]?.available && status[key]?.policy_compatible !== false).length,
    [selected, status],
  );

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0] ?? null;
    setFile(next);
    setResults([]);
    setDecision(null);
    setPreview(next ? URL.createObjectURL(next) : "");
    setMessage(next ? `${next.name} is ready for analysis.` : "Choose an image to begin.");
  }

  function toggleModel(key: ModelKey) {
    if (policy?.available) return;
    if (!status[key]?.available || status[key]?.policy_compatible === false) return;
    setSelected((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);
  }

  async function runInference(event: FormEvent) {
    event.preventDefault();
    if (!file || !selected.length) return;
    setRunning(true);
    setResults([]);
    setDecision(null);
    setMessage("Running full-resolution inference. Please wait…");
    const body = new FormData();
    body.append("image", file);
    body.append("models", selected.join(","));
    try {
      const response = await fetch(`${apiBase}/infer`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Inference failed.");
      setResults(data.results ?? []);
      setDecision(data.decision ?? null);
      setMessage(`Completed ${data.results?.length ?? 0} model run(s) with ${data.mode === "learned_hybrid_policy" ? "the fully automatic hybrid policy" : data.mode === "calibrated_policy" ? "the frozen spatial policy" : "legacy fallback"}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Inference failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main>
      <header className="site-header">
        <div className="brand"><span className="brand-mark">AS</span><span>Aluminum Surface Lab</span></div>
        <div className="header-label">Segmentation inference demo</div>
      </header>

      <section className="hero">
        <p className="eyebrow">Industrial visual inspection</p>
        <h1>Compare three defect-segmentation models on one image.</h1>
        <p className="hero-copy">Upload an aluminum-surface image. The demo returns an original-image overlay and a binary defect mask for each selected model.</p>
      </section>

      <form className="workspace" onSubmit={runInference}>
        <section className="control-panel" aria-label="Inference controls">
          <div className="panel-heading"><p className="step">01 / IMAGE</p><h2>Input image</h2></div>
          <label className="upload-zone" htmlFor="image-upload">
            <input id="image-upload" type="file" accept="image/png,image/jpeg,image/jpg" onChange={chooseImage} />
            {preview ? <img src={preview} alt="Selected input" /> : <><span className="upload-icon">↑</span><strong>Drop or select an image</strong><small>PNG or JPEG · original resolution preserved</small></>}
          </label>

          <div className="panel-heading model-heading"><p className="step">02 / MODELS</p><h2>Architecture selection</h2></div>
          <div className="model-list">
            {models.map((model) => {
              const modelState = status[model.key];
              return <label className={`model-option ${selected.includes(model.key) ? "selected" : ""}`} key={model.key}>
                <input type="checkbox" checked={selected.includes(model.key)} disabled={policy?.available || (modelState ? (!modelState.available || modelState.policy_compatible === false) : true)} onChange={() => toggleModel(model.key)} />
                <span><strong>{model.name}</strong><small>{model.subtitle}</small></span>
                <em className={modelState?.available && modelState?.policy_compatible !== false ? "ready" : "offline"}>{modelState ? (!modelState.available ? modelState.reason || "Unavailable" : modelState.policy_compatible === false ? "Policy not calibrated" : "Ready") : "Checking…"}</em>
              </label>;
            })}
          </div>

          <div className={`policy-status ${policy?.available ? "ready" : "offline"}`}>
            <strong>{policy?.available ? policy.mode === "hybrid_fully_automatic" ? "Fully automatic hybrid policy active" : "Frozen spatial policy active" : "Decision policy missing"}</strong>
            <small>{policy?.available ? `Validation-calibrated · ${(policy.models ?? []).join(" + ")} · ${policy.mode === "hybrid_fully_automatic" ? "PASS / DEFECT, no REVIEW" : `target FNR ≤ ${((policy.targets?.max_alert_fnr ?? 0) * 100).toFixed(1)}%`}${policy.ready === false ? " · one or more required models are unavailable" : ""}` : "The API will use the legacy 0.5 threshold until decision_policy.json is created."}</small>
          </div>
          <button type="submit" disabled={!file || !selected.length || availableCount !== selected.length || (policy?.available && policy.ready === false) || running}>{running ? "Running inference…" : `Run ${selected.length} model${selected.length > 1 ? "s" : ""}`}</button>
          <p className="service-note">{availableCount}/{selected.length} selected checkpoint(s) available · {message}</p>
        </section>

        <section className="results-panel" aria-live="polite">
          <div className="results-heading"><div><p className="step">03 / DECISION</p><h2>{policy?.mode === "hybrid_fully_automatic" ? "PASS / DEFECT" : "PASS / REVIEW / DEFECT"}</h2></div><span>Frozen validation policy</span></div>
          {!results.length && <div className="empty-state"><div className="grid-motif" /><p>Results will appear here after inference.</p><small>Each result includes an overlay for review and a downloadable binary mask.</small></div>}
          {decision && <article className={`decision-card ${decision.level}`}>
            <div className="decision-copy">
              <div><small>Final decision</small><strong>{decision.level.toUpperCase()}</strong></div>
              <p>{decision.reason}</p>
              <ul>
                <li>Spatial agreement: {decision.max_spatial_votes}/{decision.required_votes} required vote(s)</li>
                <li>Strong models: {decision.strong_models.join(", ") || "none"}</li>
                <li>Candidate pixels: {decision.mask_pixels.toLocaleString()}</li>
              </ul>
            </div>
            <div className="decision-visuals"><figure><img src={decision.overlay_png} alt="Final decision overlay" /><figcaption>Decision overlay</figcaption></figure><figure><img src={decision.mask_png} alt="Final decision mask" /><figcaption>Final mask</figcaption></figure></div>
          </article>}
          <div className="result-grid">
            {results.map((result) => <article className="result-card" key={result.model}>
              <div className="result-title"><div><h3>{models.find((item) => item.key === result.model)?.name ?? result.model}</h3><p>{result.elapsed_seconds.toFixed(2)} s · threshold {result.threshold.toFixed(2)} · minimum area {result.min_component_area_px}px</p></div><span className={`model-state ${result.state}`}>{result.state === "strong" ? "Strong" : result.state}</span></div>
              <div className="visuals"><figure><img src={result.overlay_png} alt={`${result.model} overlay`} /><figcaption>Defect overlay</figcaption></figure><figure><img src={result.mask_png} alt={`${result.model} binary mask`} /><figcaption>Binary mask</figcaption></figure></div>
              <div className="downloads"><a href={result.overlay_png} download={`${result.model}_overlay.png`}>Download overlay</a><a href={result.mask_png} download={`${result.model}_mask.png`}>Download mask</a></div>
            </article>)}
          </div>
        </section>
      </form>
    </main>
  );
}
