"use client";

/* Uploaded blob previews and API-returned data URLs cannot use Next image optimization. */
/* eslint-disable @next/next/no-img-element */

import { ChangeEvent, FormEvent, useEffect, useState } from "react";

type ModelKey = "unet" | "segformer" | "vmamba";
type ModeKey =
  | "raw_unet"
  | "raw_segformer"
  | "raw_vmamba"
  | "unet"
  | "segformer"
  | "vmamba"
  | "unet_segformer"
  | "unet_vmamba"
  | "segformer_vmamba";
type ModelStatus = {
  available: boolean;
  checkpoint: string;
  reason?: string;
};
type ModeStatus = {
  id: ModeKey;
  models: ModelKey[];
  available: boolean;
  models_available: boolean;
  ready: boolean;
  reason: string;
  path: string;
  targets?: Record<string, number>;
  threshold?: number;
  mode: "raw_segmentation" | "adaptive_single_model" | "spatial_pair_ensemble";
};
type InferenceMode = {
  key: ModeKey;
  group: "Original - no Adaptive" | "Adaptive single model" | "Spatial pair ensemble";
  name: string;
  subtitle: string;
  models: ModelKey[];
  strategy: "raw" | "calibrated";
};
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

const inferenceModes: InferenceMode[] = [
  { key: "raw_unet", group: "Original - no Adaptive", name: "Original U-Net", subtitle: "Raw probability mask at the frozen Validation threshold", models: ["unet"], strategy: "raw" },
  { key: "raw_segformer", group: "Original - no Adaptive", name: "Original SegFormer", subtitle: "Raw probability mask at the frozen Validation threshold", models: ["segformer"], strategy: "raw" },
  { key: "raw_vmamba", group: "Original - no Adaptive", name: "Original VMamba", subtitle: "Raw probability mask at the frozen Validation threshold", models: ["vmamba"], strategy: "raw" },
  { key: "unet", group: "Adaptive single model", name: "U-Net rule-based", subtitle: "Adaptive connected-component rules", models: ["unet"], strategy: "calibrated" },
  { key: "segformer", group: "Adaptive single model", name: "SegFormer rule-based", subtitle: "Adaptive connected-component rules", models: ["segformer"], strategy: "calibrated" },
  { key: "vmamba", group: "Adaptive single model", name: "VMamba rule-based", subtitle: "Adaptive connected-component rules", models: ["vmamba"], strategy: "calibrated" },
  { key: "unet_segformer", group: "Spatial pair ensemble", name: "U-Net + SegFormer", subtitle: "Spatial agreement between two model masks", models: ["unet", "segformer"], strategy: "calibrated" },
  { key: "unet_vmamba", group: "Spatial pair ensemble", name: "U-Net + VMamba", subtitle: "Spatial agreement between two model masks", models: ["unet", "vmamba"], strategy: "calibrated" },
  { key: "segformer_vmamba", group: "Spatial pair ensemble", name: "SegFormer + VMamba", subtitle: "Spatial agreement between two model masks", models: ["segformer", "vmamba"], strategy: "calibrated" },
];

const apiBase = process.env.NEXT_PUBLIC_INFERENCE_API ?? "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [selectedMode, setSelectedMode] = useState<ModeKey>("unet_vmamba");
  const [status, setStatus] = useState<Record<string, ModelStatus>>({});
  const [modeStatus, setModeStatus] = useState<Partial<Record<ModeKey, ModeStatus>>>({});
  const [results, setResults] = useState<Result[]>([]);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [message, setMessage] = useState("Choose an image to begin.");
  const [running, setRunning] = useState(false);

  const activeMode = inferenceModes.find((mode) => mode.key === selectedMode)
    ?? inferenceModes.find((mode) => mode.key === "unet_vmamba")!;
  const selected = activeMode.models;
  const activePolicy = modeStatus[selectedMode];
  const availableCount = selected.filter((key) => status[key]?.available).length;

  useEffect(() => {
    fetch(apiBase + "/health")
      .then((response) => {
        if (!response.ok) throw new Error("Inference service health check failed.");
        return response.json();
      })
      .then((data) => {
        const nextStatus = data.models ?? {};
        const nextModes = Object.fromEntries(
          ((data.modes ?? []) as ModeStatus[]).map((mode) => [mode.id, mode]),
        ) as Partial<Record<ModeKey, ModeStatus>>;
        const preferred = (data.default_mode as ModeKey | undefined) ?? "unet_vmamba";
        const fallback = inferenceModes.find((mode) => nextModes[mode.key]?.ready)?.key;
        setStatus(nextStatus);
        setModeStatus(nextModes);
        setSelectedMode(nextModes[preferred]?.ready ? preferred : (fallback ?? preferred));
        if (!fallback && !nextModes[preferred]?.ready) {
          setMessage("No calibrated inference mode is ready. Check checkpoints and policies.");
        }
      })
      .catch(() => setMessage("Inference service is offline. Start the local Python service first."));
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0] ?? null;
    setFile(next);
    setResults([]);
    setDecision(null);
    setPreview(next ? URL.createObjectURL(next) : "");
    setMessage(next ? next.name + " is ready for analysis." : "Choose an image to begin.");
  }

  function chooseMode(key: ModeKey) {
    if (!modeStatus[key]?.ready || running) return;
    setSelectedMode(key);
    setResults([]);
    setDecision(null);
    const chosen = inferenceModes.find((mode) => mode.key === key);
    setMessage(chosen ? chosen.name + " selected." : "Inference mode selected.");
  }

  async function runInference(event: FormEvent) {
    event.preventDefault();
    if (!file || !selected.length || !activePolicy?.ready) return;
    setRunning(true);
    setResults([]);
    setDecision(null);
    setMessage("Running full-resolution inference. Please wait...");
    const body = new FormData();
    body.append("image", file);
    body.append("models", selected.join(","));
    body.append("decision_mode", activeMode.strategy);
    try {
      const response = await fetch(apiBase + "/infer", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Inference failed.");
      setResults(data.results ?? []);
      setDecision(data.decision ?? null);
      setMessage(
        activeMode.strategy === "raw"
          ? "Completed " + activeMode.name + " without component or spatial adaptation."
          : "Completed " + activeMode.name + " with its frozen validation policy.",
      );
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
        <h1>Compare original model outputs with calibrated rule-based decisions.</h1>
        <p className="hero-copy">Choose a raw model baseline, an Adaptive single-model policy, or a two-model spatial ensemble, then inspect the final decision, overlays, and binary masks.</p>
      </section>

      <form className="workspace" onSubmit={runInference}>
        <section className="control-panel" aria-label="Inference controls">
          <div className="panel-heading"><p className="step">01 / IMAGE</p><h2>Input image</h2></div>
          <label className="upload-zone" htmlFor="image-upload">
            <input id="image-upload" type="file" accept="image/png,image/jpeg,image/jpg" onChange={chooseImage} />
            {preview ? <img src={preview} alt="Selected input" /> : <><span className="upload-icon">+</span><strong>Drop or select an image</strong><small>PNG or JPEG - original resolution preserved</small></>}
          </label>

          <div className="panel-heading model-heading"><p className="step">02 / MODE</p><h2>Inference selection</h2></div>
          <div className="model-list">
            {(["Original - no Adaptive", "Adaptive single model", "Spatial pair ensemble"] as const).map((group) => (
              <div className="mode-group" key={group}>
                <p className="mode-group-title">{group}</p>
                {inferenceModes.filter((mode) => mode.group === group).map((mode) => {
                  const current = modeStatus[mode.key];
                  const label = !current ? "Checking..." : current.ready ? "Ready" : current.reason || "Unavailable";
                  return <label className={"model-option " + (selectedMode === mode.key ? "selected" : "")} key={mode.key}>
                    <input type="radio" name="inference-mode" value={mode.key} checked={selectedMode === mode.key} disabled={!current?.ready || running} onChange={() => chooseMode(mode.key)} />
                    <span><strong>{mode.name}</strong><small>{mode.subtitle}</small></span>
                    <em className={current?.ready ? "ready" : "offline"}>{label}</em>
                  </label>;
                })}
              </div>
            ))}
          </div>

          <div className={"policy-status " + (activePolicy?.ready ? "ready" : "offline")}>
            <strong>
              {activePolicy?.ready
                ? activeMode.strategy === "raw"
                  ? activeMode.name + " baseline ready"
                  : activeMode.name + " policy ready"
                : "Selected mode unavailable"}
            </strong>
            <small>
              {activePolicy?.ready
                ? activeMode.strategy === "raw"
                  ? "Original raw segmentation - threshold " + (activePolicy.threshold ?? activePolicy.targets?.threshold ?? 0).toFixed(2) + " - no component filtering"
                  : "Validation-calibrated - " + selected.join(" + ") + " - target FNR <= " + (((activePolicy.targets?.max_alert_fnr ?? 0) * 100).toFixed(1)) + "%"
                : activePolicy?.reason ?? "Loading mode status..."}
            </small>
          </div>
          <button type="submit" disabled={!file || !activePolicy?.ready || availableCount !== selected.length || running}>{running ? "Running inference..." : "Run " + activeMode.name}</button>
          <p className="service-note">{availableCount}/{selected.length} selected checkpoint(s) available - {message}</p>
        </section>

        <section className="results-panel" aria-live="polite">
          <div className="results-heading"><div><p className="step">03 / DECISION</p><h2>{activeMode.strategy === "raw" ? "PASS / DEFECT" : "PASS / REVIEW / DEFECT"}</h2></div><span>{activeMode.group}</span></div>
          {!results.length && <div className="empty-state"><div className="grid-motif" /><p>Results will appear here after inference.</p><small>Each result includes an overlay for review and a downloadable binary mask.</small></div>}
          {decision && <article className={"decision-card " + decision.level}>
            <div className="decision-copy">
              <div><small>Final decision - {activeMode.name}</small><strong>{decision.level.toUpperCase()}</strong></div>
              <p>{decision.reason}</p>
              <ul>
                <li>Vote agreement: {decision.max_spatial_votes}/{decision.required_votes} required vote(s)</li>
                <li>Strong models: {decision.strong_models.join(", ") || "none"}</li>
                <li>Candidate pixels: {decision.mask_pixels.toLocaleString()}</li>
              </ul>
            </div>
            <div className="decision-visuals"><figure><img src={decision.overlay_png} alt="Final decision overlay" /><figcaption>Decision overlay</figcaption></figure><figure><img src={decision.mask_png} alt="Final decision mask" /><figcaption>Final mask</figcaption></figure></div>
          </article>}
          <div className="result-grid">
            {results.map((result) => <article className="result-card" key={result.model}>
              <div className="result-title"><div><h3>{models.find((item) => item.key === result.model)?.name ?? result.model}</h3><p>{result.elapsed_seconds.toFixed(2)} s - threshold {result.threshold.toFixed(2)} - minimum area {result.min_component_area_px}px</p></div><span className={"model-state " + result.state}>{result.state === "strong" ? "Strong" : result.state}</span></div>
              <div className="visuals"><figure><img src={result.overlay_png} alt={result.model + " overlay"} /><figcaption>Defect overlay</figcaption></figure><figure><img src={result.mask_png} alt={result.model + " binary mask"} /><figcaption>Binary mask</figcaption></figure></div>
              <div className="downloads"><a href={result.overlay_png} download={result.model + "_overlay.png"}>Download overlay</a><a href={result.mask_png} download={result.model + "_mask.png"}>Download mask</a></div>
            </article>)}
          </div>
        </section>
      </form>
    </main>
  );
}
