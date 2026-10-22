"use strict";

const $ = id => document.getElementById(id);
const endpoint = "/api/workbench";
let state = null;
let candidate = null;
let examples = [];
let busy = false;
let recognitionRevision = null;
const images = {source: null, reference: null};

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
function announce(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}
async function api(path, options = {}) {
  const response = await fetch(endpoint + path, options);
  let body;
  try { body = await response.json(); }
  catch (_) { throw new Error(`The server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const error = new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
    error.status = response.status;
    throw error;
  }
  return body;
}
function post(path, body) {
  return api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
}
function registry() { return state && state.registry ? state.registry : {}; }
function patchId(record) { return record.id || record.patch_id; }
function metadata(record) { return record.metadata || record; }
function patchTitle(record) { return record.title || metadata(record).title || patchId(record); }
function activeIds() { return registry().active_patch_ids || []; }
function isActive(record) { return activeIds().includes(patchId(record)) || record.active === true; }
function isApproved(record) { return record.status !== "withdrawn" && (record.status === "approved" || Boolean(record.approved_at)); }
function isStale(record) {
  const currentParent = activeIds().length ? activeIds()[activeIds().length - 1] : null;
  return (record.parent_sha256 && record.parent_sha256 !== registry().current_sha256)
    || (record.parent_patch_id ?? null) !== currentParent;
}
function count(metric) {
  if (!metric || typeof metric !== "object") return null;
  const correct = metric.correct ?? metric.numerator;
  const total = metric.total ?? metric.denominator;
  return typeof correct === "number" && typeof total === "number" ? `${correct}/${total}` : null;
}
function readable(value) {
  if (value === undefined || value === null) return "Not recorded";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
function when(value) {
  if (!value) return "Time not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
function jsonDetails(title, data) {
  const details = node("details", undefined, "json-details");
  details.append(node("summary", title), node("pre", JSON.stringify(data, null, 2)));
  return details;
}
function download(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2) + "\n"], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const link = node("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function button(text, handler, className = "secondary compact", disabled = false) {
  const result = node("button", text, className);
  result.type = "button"; result.disabled = disabled || busy;
  result.addEventListener("click", handler);
  return result;
}

function renderState() {
  const current = registry();
  if (recognitionRevision !== null && recognitionRevision !== current.revision) {
    $("recognition-result").replaceChildren(node("p", "The active model version changed. Compare the photograph again to see its current prediction.", "help"));
    recognitionRevision = null;
  }
  $("state-cards").replaceChildren();
  const cards = [
    ["Registry revision", readable(current.revision), "Every approved state change is recorded"],
    ["Active updates", activeIds().length, state.reference_count ? `${state.reference_count} reference examples available` : "Reference set size is shown in candidate evidence"],
    ["Active projection fingerprint", current.current_sha256 ? `${current.current_sha256.slice(0,23)}…` : "Not recorded", current.current_sha256 === current.base_sha256 ? "Original base projection active" : "A parameter update is active"]
  ];
  for (const [title, value, detail] of cards) {
    const card = node("div", undefined, "state-card");
    const valueNode = node("div", value, "state-value" + (title.includes("fingerprint") ? " hash" : ""));
    if (title.includes("fingerprint")) valueNode.title = current.current_sha256 || "";
    card.append(node("div", title, "state-label"), valueNode, node("div", detail, "state-detail"));
    $("state-cards").append(card);
  }
  $("restore-base").disabled = busy || current.current_sha256 === current.base_sha256;
  $("rollback-latest").disabled = busy || !activeIds().length;
  $("connection").textContent = "● Persistent registry connected";
  const referenceMetrics = state.reference_metrics || {};
  $("live-reference-metrics").replaceChildren();
  const liveGrid = node("div", undefined, "metric-grid");
  for (const [key, title] of [["clean_accuracy", "Current version / clean reference images"], ["text_accuracy", "Current version / legitimate text reference images"]]) {
    const value = count(referenceMetrics[key]);
    if (!value) continue;
    const card = node("div", undefined, "metric-card");
    card.append(node("div", title, "metric-title"), node("div", value, "metric-value"), node("div", "Correct / evaluated reference examples · development evidence", "metric-detail"));
    liveGrid.append(card);
  }
  if (liveGrid.children.length) $("live-reference-metrics").append(liveGrid);
}
function populateLabels() {
  const selected = $("expected-label").value;
  $("expected-label").replaceChildren(node("option", "Choose the actual object…"));
  $("expected-label").firstChild.value = "";
  for (const label of state.labels || []) {
    const text = typeof label === "string" ? label : label.label || label.name || label.id;
    const option = node("option", text);
    option.value = typeof label === "string" ? label : label.label || label.name || label.id;
    $("expected-label").append(option);
  }
  $("expected-label").value = selected;
}
function actionContext() {
  const actor = $("action-actor").value.trim();
  const reason = $("action-reason").value.trim();
  if (!actor || !reason) {
    $("action-actor").scrollIntoView({behavior: "smooth", block: "center"});
    (!actor ? $("action-actor") : $("action-reason")).focus();
    throw new Error("Enter the administrator identifier and a reason in the version registry before changing approval or active parameters.");
  }
  return {actor, reason};
}
function renderPatchActions(record) {
  const container = node("div", undefined, "version-actions");
  const id = patchId(record);
  if (record.status === "withdrawn") {
    container.append(node("span", "Withdrawn candidate; retained for audit only.", "help"));
    return container;
  }
  if (isActive(record)) {
    container.append(node("span", "This patch is in the active version lineage.", "help"));
    return container;
  }
  if (!isApproved(record)) {
    const approvalLabel = node("label", undefined, "approval-row");
    const approval = node("input"); approval.type = "checkbox";
    approvalLabel.append(approval, node("span", "I reviewed the evidence and approve this experimental candidate."));
    const approve = button("Approve candidate", async () => {
      if (!approval.checked) return;
      await mutate(async () => post(`/patches/${encodeURIComponent(id)}/approve`, {...actionContext(), approved: true}), "Candidate approved. Active parameters have not changed.");
    }, "secondary compact", true);
    approval.addEventListener("change", () => {approve.disabled = !approval.checked || busy;});
    const wrapper = node("div"); wrapper.append(approvalLabel, approve);
    container.append(wrapper);
  } else {
    container.append(button("Activate approved patch", async () => {
      await mutate(async () => post(`/patches/${encodeURIComponent(id)}/activate`, {...actionContext(), expected_revision: registry().revision}), "Approved candidate activated. Inspect the updated registry and audit trail.");
    }, "compact", isStale(record)));
  }
  if (isStale(record)) container.append(node("p", "Branch conflict: this candidate was fitted from another parameter state. Refresh the registry, then fit a new candidate from the active state. Activation will not replace another branch automatically.", "help bad"));
  container.append(button("Withdraw candidate", async () => {
    await mutate(async () => post(`/patches/${encodeURIComponent(id)}/withdraw`, actionContext()), "Candidate withdrawn. Its evidence and audit history remain recorded.");
  }, "secondary compact"));
  return container;
}
function renderVersions() {
  $("version-list").replaceChildren();
  const patches = state.patches || [];
  if (!patches.length) $("version-list").append(node("p", "No correction candidates have been saved. Create the first one above.", "help"));
  for (const record of [...patches].reverse()) {
    const card = node("article", undefined, "version-card");
    const top = node("div", undefined, "version-top");
    const label = record.status === "withdrawn" ? "Withdrawn" : isActive(record) ? "Active lineage" : isStale(record) ? "Branch conflict" : isApproved(record) ? "Approved" : record.status || "Candidate";
    top.append(node("div", patchTitle(record), "version-title"), node("span", label, "badge" + (isActive(record) ? " active" : isStale(record) ? " conflict" : "")));
    const meta = metadata(record);
    const identity = node("div", `${patchId(record)} · ${when(record.created_at)}`, "version-id");
    const facts = node("div", undefined, "version-meta");
    for (const [name, value] of [["Object", meta.expected_label], ["Printed word", meta.target_word], ["Author", meta.actor], ["Source", meta.source]]) {
      if (value) facts.append(node("span", `${name}: ${value}`));
    }
    card.append(top, identity, facts);
    if (record.preview) card.append(button("Inspect saved evidence", () => {candidate = {patch_id: patchId(record), record, preview: record.preview, warnings: record.warnings || []}; renderCandidate(); $("candidate").scrollIntoView({behavior: "smooth"});}));
    card.append(renderPatchActions(record), jsonDetails("Version metadata", record));
    $("version-list").append(card);
  }
}
function renderCandidate() {
  if (!candidate) return;
  const record = (state.patches || []).find(item => patchId(item) === candidate.patch_id) || candidate.record || candidate;
  const preview = candidate.preview || record.preview || {};
  const content = $("candidate-content"); content.className = ""; content.replaceChildren();
  const heading = node("div", undefined, "candidate-heading");
  heading.append(node("h3", patchTitle(record)), node("span", record.status === "withdrawn" ? "Withdrawn" : isActive(record) ? "Active lineage" : isApproved(record) ? "Approved" : "Candidate", "badge"));
  content.append(heading, node("p", `Candidate ${candidate.patch_id || patchId(record)}. Measurements below belong to this candidate; they are not automatically final test results.`, "help"));
  const support = preview.support || {};
  if (support.before_label !== undefined || support.after_label !== undefined) {
    const result = node("div", undefined, "support-result");
    const description = node("div");
    description.append(node("div", "Teaching image / object recognition", "support-label"));
    const predictions = node("div", undefined, "before-after");
    predictions.append(node("b", readable(support.before_label)), node("span", "→"), node("b", readable(support.after_label)));
    description.append(predictions);
    result.append(description, node("div", `Expected: ${support.expected_label || metadata(record).expected_label || "Not recorded"}`, "help"));
    content.append(result);
  }
  const metrics = preview.metrics || {};
  const metricGrid = node("div", undefined, "metric-grid");
  for (const [key, title] of [["clean_accuracy", "Reference set / clean object recognition"], ["text_accuracy", "Reference set / legitimate text recognition"], ["attacked_accuracy", "Reference set / misleading-label recognition"], ["distractor_accuracy", "Reference set / misleading-label recognition"]]) {
    const metric = metrics[key];
    if (!metric) continue;
    const before = count(metric.before), after = count(metric.after);
    if (!before || !after) continue;
    const card = node("div", undefined, "metric-card");
    card.append(node("div", title, "metric-title"), node("div", `${before} → ${after}`, "metric-value"), node("div", "Before → candidate · correct / evaluated examples", "metric-detail"));
    metricGrid.append(card);
  }
  if (metricGrid.children.length) content.append(metricGrid);
  else content.append(node("p", "Structured before/after reference-set counts were not supplied for this candidate. Inspect the full evidence below before approving.", "help"));
  if (preview.new_errors !== undefined) {
    const newErrors = node("div", undefined, "warning-list");
    newErrors.append(node("strong", "Collateral errors on the supplied reference set"), node("p", readable(preview.new_errors)), node("p", "A zero count only covers the evaluated reference examples; it does not guarantee preservation on other images."));
    content.append(newErrors);
  }
  const warnings = candidate.warnings || preview.warnings || record.warnings || [];
  if (warnings.length) {
    const box = node("div", undefined, "warning-list");
    box.append(node("strong", "Review limitations"));
    for (const warning of warnings) box.append(node("p", readable(warning)));
    content.append(box);
  }
  content.append(node("p", "These reference checks may guide your decision, so they are development evidence. Independent frozen evaluation must use source images excluded from correction fitting and selection.", "help"));
  content.append(jsonDetails("Full candidate evidence and measurements", candidate));
  const review = node("div", undefined, "candidate-review");
  review.append(node("strong", "Approval does not activate the patch."), node("p", "Administrator identity and decision reason are recorded in the version registry below. Activation is a separate action with a revision check.", "help"), renderPatchActions(record));
  content.append(review);
  $("export-candidate").disabled = false;
}
function renderHistory(events) {
  $("event-list").replaceChildren(); $("event-count").textContent = `${events.length} events`;
  if (!events.length) $("event-list").append(node("p", "No registry changes have been recorded.", "help"));
  for (const event of [...events].reverse()) {
    const row = node("div", undefined, "event");
    const description = node("div");
    description.append(node("strong", `${event.action} · ${event.actor || "Actor not recorded"}`), node("div", `${event.reason || "Reason not recorded"}${event.patch_id ? ` · ${event.patch_id}` : ""}${event.revision !== undefined ? ` · revision ${event.revision}` : ""}`, "event-description"));
    row.append(description, node("span", when(event.at), "event-time"));
    $("event-list").append(row);
  }
}
function studyTable(headers, rows, className = "") {
  const wrapper = node("div", undefined, "study-table-wrap " + className);
  const table = node("table"), head = node("thead"), body = node("tbody"), heading = node("tr");
  for (const title of headers) heading.append(node("th", title));
  head.append(heading);
  for (const values of rows) {
    const row = node("tr");
    for (const value of values) row.append(node("td", value));
    body.append(row);
  }
  table.append(head, body); wrapper.append(table);
  return wrapper;
}
function studyMetric(metric) {
  const value = count(metric);
  if (!value) return "Not recorded";
  const denominator = metric.total ?? metric.denominator;
  return denominator === 0 ? `${value} (undefined rate)` : value;
}
function studyMethod(method) {
  return ({base: "Unchanged CLIP", object_only: "Object-only projection edit", "object-only": "Object-only projection edit", protected: "Text-protected projection edit", finetune: "Projection fine-tuning", oracle_mask: "Oracle masking (privileged)"})[method] || method;
}
function studyResultRows(results) {
  const rows = [];
  for (const [word, methods] of Object.entries(results)) {
    for (const [method, summary] of Object.entries(methods)) {
      const textPreservation = summary.preservation && summary.preservation.legitimate_text;
      rows.push([word, studyMethod(method), studyMetric(summary.target_distractor_accuracy),
        studyMetric(summary.clean_object_accuracy), studyMetric(summary.legitimate_text_accuracy),
        studyMetric(summary.other_word_distractor_accuracy),
        studyMetric(textPreservation && textPreservation.baseline_correct_preserved),
        studyMetric(textPreservation && textPreservation.new_errors)]);
    }
  }
  return rows;
}
function studyFigures(content) {
  const figures = node("div", undefined, "study-figures");
  for (const [filename, title, description] of [
    ["selectivity.png", "Independent edit selectivity", "Held-out target repair, clean recognition and legitimate text recognition for each method and edit word"],
    ["interference.png", "Sequential editing and interference", "Held-out recognition after all three edits, compared across the six application orders"]
  ]) {
    const figure = node("figure"); figure.hidden = true;
    const image = node("img"); image.alt = description;
    image.addEventListener("load", () => {figure.hidden = false;});
    const caption = node("figcaption");
    const link = node("a", title + " ↗"); link.href = `${endpoint}/figures/${filename}`; link.target = "_blank"; link.rel = "noopener";
    caption.append(link); figure.append(image, caption);
    image.src = `${endpoint}/figures/${filename}`;
    figures.append(figure);
  }
  content.append(figures);
}
function renderStudy(report) {
  const content = $("study-content"); content.replaceChildren();
  if (!report || report.status === "not_run" || report.available === false) {
    content.append(node("p", "The independent selection study has not run. Candidate previews remain development evidence; no independent accuracy is claimed here.", "help"));
    return;
  }
  if (report.description || report.question) content.append(node("p", report.question || report.description, "muted"));
  const isFinal = report.status === "completed_frozen_selectivity_evaluation";
  const isDevelopment = Boolean(report.validation_results) && !isFinal;
  content.append(node("span", isFinal ? "Frozen held-out evaluation completed" : isDevelopment ? "Development / validation selection only" : `Recorded status: ${report.status || "not specified"}`, "badge"));
  const dataset = report.dataset;
  if (dataset) {
    const summary = node("div", undefined, "study-dataset-summary");
    for (const [name, value] of [["Held-out source photographs", dataset.final_photos], ["Object questions", dataset.object_questions], ["Text questions", dataset.text_questions]]) {
      if (value === undefined) continue;
      const item = node("div"); item.append(node("strong", value), node("span", name)); summary.append(item);
    }
    content.append(summary);
    content.append(node("p", `${dataset.denominator_policy || "All recorded questions remain in the denominators."} ${dataset.human_reviewed === false ? "Labels received automated auditing; independent human label review was not performed." : ""}`, "help"));
  }
  if (report.selection) {
    const selected = [];
    for (const [method, selection] of Object.entries(report.selection)) {
      if (selection && selection.selected) selected.push(`${studyMethod(method)}: ${selection.selected}`);
    }
    if (selected.length) content.append(node("p", `Selected on validation before final inference — ${selected.join("; ")}.`, "help"));
  }
  const independentResults = isFinal ? report.results : isDevelopment ? report.validation_results : null;
  if (independentResults) {
    content.append(node("h3", isFinal ? "Independent edits / held-out results" : "Development candidates / validation results", "study-subheading"));
    if (isDevelopment) {
      const warning = node("div", undefined, "warning-list");
      warning.append(node("strong", "Final evaluation has not run."), node("p", "These validation measurements were used to select settings. They are development evidence and cannot be reported as independent final performance."));
      content.append(warning);
    }
    content.append(node("p", "Accuracy columns show correct / all evaluated questions. Text preservation shows retained correct answers / originally correct text questions. New text errors use that same originally correct denominator; they are not a second accuracy measure. Clean and text tasks repeat across edit words because each independently edited model is evaluated on them.", "help"));
    content.append(studyTable(["Edit word", "Method", "Target-label images", "Clean objects", "Legitimate text", "Other-word images", "Text preservation", "New text errors"], studyResultRows(independentResults), "independent-results"));
    content.append(node("p", "Oracle masking knows the synthetic label location and which task is being asked. Its privileged information is unavailable to the learned edits. Legitimate text cards test a controlled reading task, rather than natural-scene OCR.", "help"));
  }
  if (Array.isArray(report.sequential) && report.sequential.length) {
    content.append(node("h3", isFinal ? "Sequential edits / all six application orders" : "Sequential artifact and rollback checks", "study-subheading"));
    if (isFinal) {
      const words = Object.keys(report.results || {});
      const rows = [];
      for (const sequence of report.sequential) {
        const finalStage = sequence.stages && sequence.stages[sequence.stages.length - 1];
        const summaries = finalStage && finalStage.results || {};
        const common = summaries[words[0]] || {};
        rows.push([(sequence.order || []).join(" → "), ...words.map(word => studyMetric(summaries[word] && summaries[word].target_distractor_accuracy)),
          studyMetric(common.clean_object_accuracy), studyMetric(common.legitimate_text_accuracy),
          sequence.exact_rollback === true ? "Exact" : sequence.exact_rollback === false ? "Failed" : "Not recorded"]);
      }
      content.append(node("p", "Each row reports the final state after all three edits, in that order. Target columns use each word's own attacked-image questions. Clean and text columns evaluate the shared tasks once per final state; the counts are not summed across words or orders. Intermediate-stage details remain in the exported study JSON.", "help"));
      content.append(studyTable(["Application order", ...words.map(word => `${word} label`), "Clean objects", "Legitimate text", "Parameter rollback"], rows));
    } else {
      content.append(node("p", "All declared order artifacts may be fitted and their parameter restoration checked during development. Held-out sequential accuracy is available only after frozen final inference.", "help"));
      content.append(studyTable(["Application order", "Saved stages", "Exact parameter rollback"], report.sequential.map(sequence => [
        (sequence.order || []).join(" → "), (sequence.stages || []).length,
        sequence.exact_rollback === true ? "Exact" : sequence.exact_rollback === false ? "Failed" : "Not recorded"
      ])));
    }
  }
  if (isFinal) studyFigures(content);
  if (report.paired_uncertainty_vs_base) content.append(jsonDetails("Paired uncertainty versus unchanged CLIP / grouped by source", report.paired_uncertainty_vs_base));
  if (report.limitations) {
    const limitations = Array.isArray(report.limitations) ? report.limitations : [report.limitations];
    const box = node("div", undefined, "warning-list"); box.append(node("strong", "Scope and limitations"));
    for (const limitation of limitations) box.append(node("p", readable(limitation)));
    content.append(box);
  }
  content.append(jsonDetails("Full recorded study and denominators", report));
}

async function refresh() {
  const [statusResult, patchesResult, historyResult] = await Promise.allSettled([api("/status"), api("/patches"), api("/history")]);
  if (statusResult.status !== "fulfilled") throw statusResult.reason;
  state = statusResult.value;
  if (patchesResult.status === "fulfilled") state.patches = patchesResult.value.patches || [];
  else if (!state.patches) throw patchesResult.reason;
  populateLabels(); renderState(); renderVersions(); renderCandidate();
  if (historyResult.status === "fulfilled") renderHistory(historyResult.value.events || []);
  else $("event-list").textContent = `Audit history could not be loaded: ${historyResult.reason.message}`;
  if (state.study || state.selection_study) renderStudy(state.study || state.selection_study);
}
async function mutate(operation, success) {
  if (busy) return;
  try {
    actionContext(); busy = true; renderState(); renderVersions(); renderCandidate();
    await operation();
    await refresh(); announce(success);
  } catch (error) {
    if (error.status === 409) {
      try { await refresh(); } catch (_) { /* Keep the original conflict visible. */ }
      announce(`State conflict: ${error.message} The registry has been refreshed. Review the active branch; no automatic replacement was attempted.`, true);
    } else announce(error.message, true);
  } finally {
    busy = false;
    if (state) {renderState(); renderVersions(); renderCandidate();}
  }
}
function readFile(file) {
  if (!file) return Promise.resolve(null);
  if (file.size > 5 * 1024 * 1024) return Promise.reject(new Error("Each image must be 5 MB or smaller."));
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) return Promise.reject(new Error("Choose a PNG, JPEG or WebP image."));
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("The selected image could not be read."));
    reader.readAsDataURL(file);
  });
}
function setImage(key, data, name) {
  images[key] = data;
  const preview = node("img"); preview.src = data;
  preview.alt = key === "source" ? "Selected image with misleading association" : "Selected matched reference image";
  $(key + "-preview").replaceChildren(preview);
  $(key + "-name").textContent = name;
}
for (const key of ["source", "reference"]) {
  $(key + "-file").addEventListener("change", async event => {
    try {
      const file = event.target.files[0];
      if (!file) return;
      const data = await readFile(file); setImage(key, data, file.name);
    } catch (error) {
      event.target.value = ""; images[key] = null;
      $(key + "-preview").replaceChildren(node("span", "Image rejected. Choose a PNG, JPEG or WebP up to 5 MB."));
      $(key + "-name").textContent = "No image selected";
      announce(error.message, true);
    }
  });
}
$("correction-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (busy) return;
  if (!images.source || !images.reference) {announce("Choose both the misleading image and its matched reference before fitting a candidate.", true); return;}
  const originalText = $("fit").textContent;
  try {
    busy = true; $("fit").disabled = true; $("fit").textContent = "Fitting & evaluating…";
    announce("Fitting a local projection candidate and evaluating the reference set. Active parameters will stay unchanged.");
    const actor = $("actor").value.trim(), reason = $("reason").value.trim();
    candidate = await post("/preview", {title: $("title").value.trim(), target_word: $("target-word").value.trim(), expected_label: $("expected-label").value, source_image_base64: images.source, reference_image_base64: images.reference, actor, source: $("source").value.trim(), reason});
    $("action-actor").value = actor; $("action-reason").value = reason;
    await refresh(); renderCandidate();
    $("candidate").scrollIntoView({behavior: "smooth", block: "start"});
    announce("Candidate saved and previewed. Review its evidence and any regressions. Approval and activation remain separate decisions.");
  } catch (error) {announce(error.message, true);}
  finally {busy = false; $("fit").disabled = false; $("fit").textContent = originalText; if (state) {renderState(); renderVersions(); renderCandidate();}}
});
$("export-candidate").addEventListener("click", () => {if (candidate) download(`visionpatch-${candidate.patch_id || "candidate"}.json`, candidate);});
$("refresh").addEventListener("click", async () => {try {await refresh(); announce("Registry refreshed from persistent storage.");} catch (error) {announce(error.message, true);}});
$("rollback-latest").addEventListener("click", () => mutate(() => post("/rollback", {...actionContext(), expected_revision: registry().revision, to_patch_id: null}), "Latest patch rolled back. The saved parent parameter state is active."));
$("restore-base").addEventListener("click", () => mutate(() => post("/rollback", {...actionContext(), expected_revision: registry().revision, to_patch_id: "base"}), "Original base projection restored. Previous candidates and audit history remain recorded."));
async function recognize(useSource) {
  if (busy) return;
  try {
    const data = useSource ? images.source : await readFile($("recognition-file").files[0]);
    if (!data) {announce(useSource ? "Select a correction source image above first." : "Choose a photograph for live recognition first.", true); return;}
    $("recognize").disabled = true; $("recognize-source").disabled = true;
    $("recognition-result").replaceChildren(node("p", "Running the frozen encoder with original and current projection parameters…", "help"));
    const result = await post("/predict", {image_base64: data});
    recognitionRevision = result.registry && result.registry.revision;
    if (state && recognitionRevision !== registry().revision) {
      $("recognition-result").replaceChildren(node("p", "The model version changed while recognition was running. Compare the photograph again.", "help"));
      recognitionRevision = null;
      return;
    }
    const box = node("div", undefined, "live-comparison");
    const image = node("img"); image.src = data; image.alt = "Photograph supplied for live recognition";
    const detail = node("div"), labels = node("div", undefined, "live-labels");
    for (const [name, prediction] of [["Original projection", result.before], ["Current active projection", result.current]]) {
      if (labels.children.length) labels.append(node("span", "→", "live-arrow"));
      const label = node("div"); label.append(node("small", name), node("strong", prediction ? prediction.label : "Not returned")); labels.append(label);
    }
    detail.append(labels, node("p", `Registry revision ${readable(result.registry && result.registry.revision)} · ${result.device || "device not recorded"}. The expected category was not provided to inference.`, "help"), node("p", "This photograph has no independently reviewed ground truth in this request. A changed prediction alone does not prove a successful correction.", "help"), jsonDetails("Relative label scores and parameter state", result));
    box.append(image, detail); $("recognition-result").replaceChildren(box);
  } catch (error) {$("recognition-result").replaceChildren(node("p", error.message, "help bad"));}
  finally {$("recognize").disabled = false; $("recognize-source").disabled = false;}
}
$("recognize").addEventListener("click", () => recognize(false));
$("recognize-source").addEventListener("click", () => recognize(true));
async function loadExamples() {
  try {
    const result = await api("/examples"); examples = Array.isArray(result) ? result : result.examples || [];
    if (!examples.length) return;
    for (const example of examples) {const option = node("option", `${example.label} · misleading “${example.target_word}”`); option.value = example.id; $("example").append(option);}
    $("example-section").hidden = false;
  } catch (_) { /* Provided examples are optional; uploaded pairs remain available. */ }
}
$("load-example").addEventListener("click", async () => {
  const example = examples.find(item => String(item.id) === $("example").value);
  if (!example) {announce("Choose a provided development pair first.", true); return;}
  try {
    $("load-example").disabled = true;
    async function fetchImage(url) {
      const parsed = new URL(url, location.origin);
      if (parsed.origin !== location.origin) throw new Error("Example images must come from this local server.");
      const response = await fetch(parsed.href);
      if (!response.ok) throw new Error("The example image could not be loaded.");
      const blob = await response.blob(); return readFile(blob);
    }
    const pair = await Promise.all([fetchImage(example.source_url), fetchImage(example.reference_url)]);
    setImage("source", pair[0], "Provided development source"); setImage("reference", pair[1], "Provided matched reference");
    $("title").value = `Preserve ${example.label} under a misleading ${example.target_word} label`;
    $("expected-label").value = example.label; $("target-word").value = example.target_word;
    $("source").value = example.source || "Provided development support pair";
    announce("Development pair loaded. Review the images, add your identity and reason, then fit a candidate.");
  } catch (error) {announce(error.message, true);}
  finally {$("load-example").disabled = false;}
});

(async () => {
  try {
    await refresh();
    announce("Ready to author a correction. Preview creates a saved candidate; active parameters change only after approval and activation.");
    await Promise.allSettled([loadExamples(), (async () => {
      try {const study = await api("/study"); renderStudy(study.report === undefined ? study : study.report);}
      catch (error) {$("study-content").textContent = `Study information could not be loaded: ${error.message}`;}
    })()]);
  } catch (error) {$("connection").textContent = "Connection unavailable"; announce(error.message, true);}
})();
