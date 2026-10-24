"use strict";
const $ = id => document.getElementById(id);
const methods = {mend:"Published MEND editor (adapted labels)", ft_last_layer:"Last-layer fine-tuning (15 steps, adapted baseline)"};
let csrf = "", busy = false, remoteBusy = false, ready = false, active = null, candidate = null, upload = null, objectURL = null, fixtures = [];
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-EditGuard-Token":csrf}, body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}
function message(error) { $("error").hidden = !error; $("error").textContent = error || ""; }
function controls() {
  $("generate").disabled = busy || remoteBusy || !ready;
  $("apply").disabled = busy || remoteBusy || !ready || !!active;
  $("rollback").disabled = busy || remoteBusy || !ready || !active;
  $("activate").disabled = busy || remoteBusy || !ready || !!active || !candidate;
  for (const button of document.querySelectorAll("[data-activate-candidate]")) button.disabled = busy || remoteBusy || !ready || !!active;
  for (const id of ["fixture", "upload", "question", "target", "method"]) $(id).disabled = busy || remoteBusy;
}
function resetAnswers() {
  candidate = null; $("candidate-warning").hidden = true; $("probe-summary").hidden = true;
  $("before-answer").textContent = "Generate an answer for this image and question.";
  $("after-answer").textContent = "No output generated for this query yet.";
  $("before-answer").classList.add("empty"); $("after-answer").classList.add("empty");
  $("operation-meta").textContent = "The query has changed. Generate again; earlier outputs remain in saved edit history. Model-generated text is not verified.";
  controls();
}
function text(id, value) { $(id).textContent = value === "" ? "[Empty generated answer]" : value; $(id).classList.remove("empty"); }
async function status() {
  try {
    const state = await api("api/status"); csrf = state.csrf_token; ready = state.phase === "ready";
    active = state.active_edit_id; remoteBusy = state.busy; $("status").textContent = state.busy ? "Model working…" : ready ? "●  Model ready" : state.phase === "failed" ? "Model unavailable" : "Model loading…";
    $("status").className = "badge " + (ready ? "ready" : state.phase === "failed" ? "failed" : "");
    if (state.error) message(state.error); controls();
  } catch (error) { ready = false; $("status").textContent = "Connection unavailable"; controls(); }
}
function imagePayload() { return upload ? {image_base64:upload} : {fixture_id:$("fixture").value}; }
async function perform(operation) {
  if (operation !== "rollback" && !$("question").value.trim()) return message("Enter a question first.");
  if (operation === "edit" && !$("target").value.trim()) return message("Enter the desired short answer first.");
  busy = true; controls(); message(""); $("operation-meta").textContent = operation === "edit" ? "Generating baseline and probe answers, fitting a candidate, saving and testing it, then restoring original weights…" : operation === "rollback" ? "Restoring saved original tensors and checking their exact hashes…" : "Generating from the current model weights…";
  try {
    const result = await api("api/" + operation, {question:$("question").value.trim(), target:$("target").value.trim(), method:$("method").value, ...imagePayload()});
    if (operation === "generate") {
      candidate = null; $("candidate-warning").hidden = true; $("probe-summary").hidden = true;
      const edited = !!result.active_edit_id;
      text(edited ? "after-answer" : "before-answer", result.answer);
      const other = $(edited ? "before-answer" : "after-answer");
      other.textContent = edited ? "Original model answer was not generated for this query." : "Apply a correction to compare.";
      other.classList.add("empty");
      $("operation-meta").textContent = `Generated in ${result.seconds.toFixed(2)} s · ${edited ? methods[result.active_method] || "Active edited model" : "Original model"}${edited ? " · Edit " + result.active_edit_id : ""} · Unverified output`;
    } else if (operation === "edit") {
      candidate = result.edit.id;
      text("before-answer", result.edit.before_answer); text("after-answer", result.edit.after_answer);
      const probe = result.edit.unrelated_probe || {};
      const warnings = [result.edit.direct_whole_answer_match ? "The generated answer matches the intended short target under the fixed whole-answer rule." : "Warning: the generated answer does not match the intended target."];
      if (probe.changed) warnings.push("Warning: the candidate changed the fixed unrelated probe answer.");
      warnings.push("Original weights were restored after preview. Approval is an operator research decision, not factual verification.");
      $("candidate-warning").textContent = warnings.join(" "); $("candidate-warning").hidden = false;
      $("probe-summary").textContent = probe.available ? `Fixed unrelated probe: ${probe.question} · Before: ${probe.before_answer || "[empty]"} · Candidate: ${probe.after_answer || "[empty]"}. One probe cannot establish factual correctness or general scope safety; uploads use this same fixture probe.` : probe.limitation;
      $("probe-summary").hidden = false;
      $("operation-meta").textContent = `${result.edit.method_label} · Inactive candidate saved · Exact preview restoration: ${result.edit.preview_exact_restoration ? "passed" : "failed"} · ${result.edit.seconds.toFixed(2)} s for the complete workflow`;
    } else {
      $("operation-meta").textContent = "Original editable tensors restored exactly; hashes match. Generate again to inspect restored behavior.";
      $("after-answer").textContent = "Edit rolled back."; $("after-answer").classList.add("empty");
    }
    await history();
  } catch(error) { message(error.message); await history(); }
  finally { busy = false; await status(); controls(); }
}
async function activateCandidate(editId) {
  if (!editId) return;
  busy = true; controls(); message("");
  try {
    const result = await api("api/activate", {edit_id:editId, approve:true});
    resetAnswers();
    $("operation-meta").textContent = `Approved and activated edit ${result.edit.id} · ${result.edit.method_label}. Generate a query to inspect the active model. Operator approval does not verify its answers.`;
    await history();
  } catch (error) { message(error.message); }
  finally { busy = false; await status(); controls(); }
}
function selectFixture() {
  upload = null; $("upload").value = "";
  if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null;
  const item = fixtures.find(row => row.id === $("fixture").value);
  if (item) { $("preview").src = item.image_url; $("question").value = item.question; }
  $("image-source").textContent = "Author-provided training fixture · not a held-out evaluation.";
  resetAnswers();
}
function metric(value, decimals = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "Unavailable";
}
function fraction(numerator, denominator) { return metric(numerator) + " / " + metric(denominator); }
function table(headers, rows) {
  const wrapper = document.createElement("div"); wrapper.className = "table-scroll"; wrapper.tabIndex = 0;
  const result = document.createElement("table"), head = document.createElement("thead"), header = document.createElement("tr"), body = document.createElement("tbody");
  for (const label of headers) { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = label; header.append(cell); }
  head.append(header);
  for (const values of rows) { const row = document.createElement("tr"); for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); } body.append(row); }
  result.append(head, body); wrapper.append(result); return wrapper;
}
const evaluationMethodLabel = key => ({base:"Unchanged base",prompt_only:"Prompt-only correction",updated_rag:"Updated image + text RAG",mend_context:"MEND (context aligned)",ft_last_layer:"Last-layer fine-tuning"})[key] || key;
const evaluationCategoryLabel = key => ({direct:"Direct question",paraphrase:"Held-out paraphrase",alternate_image:"Alternate image",text_locality:"Text locality",visual_locality:"Visual locality"})[key] || key.replaceAll("_", " ");
function evaluationElement(tag, className, value) {
  const element = document.createElement(tag); if (className) element.className = className;
  if (value !== undefined) element.textContent = value; return element;
}
function percentage(value) { return typeof value === "number" && Number.isFinite(value) ? metric(value * 100, 1) + "%" : "Unavailable"; }
function sumMetric(values, key) {
  return values.length && values.every(row => typeof row[key] === "number" && Number.isFinite(row[key])) ? values.reduce((sum, row) => sum + row[key], 0) : undefined;
}
function evaluationCard(label, value, description, className = "") {
  const card = evaluationElement("article", "evaluation-card " + className);
  card.append(evaluationElement("span", "evaluation-card-label", label), evaluationElement("strong", "evaluation-card-value", value), evaluationElement("p", "evaluation-card-note", description)); return card;
}
function evaluationHeading(title, description) {
  const heading = evaluationElement("div", "evaluation-heading"); heading.append(evaluationElement("h3", "", title));
  if (description) heading.append(evaluationElement("p", "muted", description)); return heading;
}
function evaluationOverview(report) {
  const scores = Object.values(report.aggregate.scores).flatMap(categories => Object.values(categories));
  const totalUnits = typeof report.completed_edit_units === "number" && typeof report.unexecuted_edit_units === "number" ? report.completed_edit_units + report.unexecuted_edit_units : undefined;
  const cards = evaluationElement("div", "evaluation-cards");
  cards.append(
    evaluationCard("Completed edit units", fraction(report.completed_edit_units, totalUnits), metric(report.unexecuted_edit_units) + " unexecuted units; each unit compares the same methods and questions."),
    evaluationCard("Assigned query instances", metric(sumMetric(scores, "denominator")), "Sum of all method × question-type denominators; includes unanswered and failed instances."),
    evaluationCard("Errors / unexecuted queries", metric(sumMetric(scores, "errors_or_unexecuted")), "Operational failures stay in every accuracy denominator.", "evaluation-card-caution"),
    evaluationCard("Empty-answer abstentions", metric(sumMetric(scores, "abstentions")), "Abstentions are counted as unsuccessful factual answers.")
  ); return cards;
}
function evaluationAccuracy(scores) {
  const section = evaluationElement("section", "evaluation-block");
  section.append(evaluationHeading("Does the correction transfer?", "Free-generated whole-answer accuracy, shown separately for the edited question, different wording and another image. Counts include errors and abstentions; they are not teacher-forced token scores."));
  const chart = evaluationElement("div", "accuracy-chart");
  for (const [method, categories] of Object.entries(scores)) {
    const row = evaluationElement("article", "accuracy-method"); row.append(evaluationElement("h4", "", evaluationMethodLabel(method)));
    const values = evaluationElement("div", "accuracy-values");
    for (const category of ["direct", "paraphrase", "alternate_image"]) {
      const score = categories[category] || {}, item = evaluationElement("div", "accuracy-item accuracy-" + category);
      const label = evaluationElement("div", "accuracy-label"); label.append(evaluationElement("span", "", evaluationCategoryLabel(category)), evaluationElement("strong", "", percentage(score.accuracy)));
      const track = evaluationElement("div", "accuracy-track"), fill = evaluationElement("div", "accuracy-fill");
      if (typeof score.accuracy === "number" && Number.isFinite(score.accuracy)) fill.style.width = Math.max(0, Math.min(100, score.accuracy * 100)) + "%";
      track.setAttribute("aria-hidden", "true"); track.append(fill);
      item.append(label, track, evaluationElement("span", "accuracy-count", fraction(score.correct, score.denominator) + " correct")); values.append(item);
    }
    row.append(values); chart.append(row);
  }
  section.append(chart); return section;
}
function evaluationRouting(retrieval) {
  if (!retrieval) return null;
  const section = evaluationElement("section", "evaluation-block");
  section.append(evaluationHeading("Updated RAG: correction routing", "Retrieval decides whether to show the current correction to the generator. Correct routing and correct generated answers are different measurements. Similarity scores are not probabilities."));
  const panels = evaluationElement("div", "evaluation-two-column");
  panels.append(table(["Ground-truth scope", "Accepted", "Rejected", "Total"], [
    ["Applicable correction", metric(retrieval.tp) + " TP", metric(retrieval.fn) + " FN", metric(retrieval.applicable_denominator)],
    ["Out of scope", metric(retrieval.fp) + " FP", metric(retrieval.tn) + " TN", metric(retrieval.out_of_scope_denominator)]
  ]));
  const accepted = typeof retrieval.tp === "number" && typeof retrieval.fp === "number" ? retrieval.tp + retrieval.fp : undefined;
  panels.append(table(["Routing measure", "Count", "Rate"], [
    ["Applicable-edit recall", fraction(retrieval.tp, retrieval.applicable_denominator), percentage(retrieval.applicable_recall)],
    ["Accepted-edit precision", fraction(retrieval.tp, accepted), percentage(retrieval.precision)],
    ["False correction application", fraction(retrieval.fp, retrieval.out_of_scope_denominator), percentage(retrieval.false_application_rate)],
    ["Operationally correct routing", fraction(retrieval.operational_correct_routing, retrieval.denominator), percentage(retrieval.operational_routing_accuracy)]
  ]));
  section.append(panels, evaluationElement("p", "muted", "Routing errors / unexecuted: " + metric(retrieval.routing_errors_or_unexecuted) + ". " + (retrieval.missing_prediction_rule || "Missing-route handling was not supplied by the report.")));
  return section;
}
function evaluationResources(report) {
  const aggregate = report.aggregate, worker = report.worker_report || {}, resources = worker.resources || {}, gpu = resources.gpu || {};
  const gib = bytes => typeof bytes === "number" && Number.isFinite(bytes) ? metric(bytes / 1073741824, 2) + " GiB" : "Unavailable";
  const section = evaluationElement("section", "evaluation-block");
  section.append(evaluationHeading("Editing cost and measured resources", "Model loading, per-edit updates and retrieval indexing are measured separately from answer latency. Each timing reports its sample count."));
  const cards = evaluationElement("div", "evaluation-cards");
  cards.append(
    evaluationCard("Model + retriever load", typeof worker.load_seconds === "number" ? metric(worker.load_seconds, 2) + " s" : "Unavailable", typeof worker.load_seconds === "number" ? "One worker load measurement; excluded from query timings." : "No worker load measurement supplied."),
    evaluationCard("Sampled host peak", gib(resources.sampled_peak_process_rss_bytes), typeof resources.host_sampling_interval_seconds === "number" ? "Process RSS sampled every " + metric(resources.host_sampling_interval_seconds * 1000) + " ms; short spikes can be missed." : "Process RSS; sampling interval unavailable."),
    evaluationCard("Peak CUDA allocation", gib(gpu.peak_allocated_bytes), "Peak tensor memory reported by the PyTorch allocator."),
    evaluationCard("Peak CUDA reservation", gib(gpu.peak_reserved_bytes), "Allocator reservation; does not include every driver allocation.")
  );
  const rows = Object.entries(aggregate.edit_latency || {}).map(([method, timing]) => [evaluationMethodLabel(method) + " update", metric(timing.sample_count), metric(timing.p50_seconds, 3), metric(timing.p95_seconds, 3)]);
  if (aggregate.rag_index_latency) { const timing = aggregate.rag_index_latency; rows.push(["Updated RAG correction indexing", metric(timing.sample_count), metric(timing.p50_seconds, 3), metric(timing.p95_seconds, 3)]); }
  section.append(cards);
  if (rows.length) section.append(table(["Operation", "Timed n", "p50 (s)", "p95 (s)"], rows));
  section.append(evaluationElement("p", "muted", "Device: " + (gpu.device || "Unavailable") + " · Worker elapsed: " + (typeof resources.elapsed_seconds === "number" ? metric(resources.elapsed_seconds, 2) + " s" : "Unavailable") + ". Worker elapsed includes loading and all evaluation operations; it is not pure training time."));
  if (resources.limits) section.append(evaluationElement("p", "muted", resources.limits));
  return section;
}
function evaluationUncertainty(paired) {
  if (!Array.isArray(paired) || !paired.length) return null;
  const details = evaluationElement("details", "evaluation-details");
  details.append(evaluationElement("summary", "", "Inspect paired uncertainty · " + paired.length + " method × question-type comparisons"), evaluationElement("p", "muted", "Differences are first method minus second method, in percentage points. Bootstrap resampling pairs the same edit across methods. With only 20 edit units, intervals may be wide; they do not establish performance on arbitrary edits."));
  const pp = value => typeof value === "number" && Number.isFinite(value) ? (value > 0 ? "+" : "") + metric(value * 100, 1) : "Unavailable";
  details.append(table(["First method", "Second method", "Question type", "Difference (pp)", "Paired 95% interval (pp)", "Edit units", "Resamples"], paired.map(row => {
    const interval = row.paired_edit_bootstrap_95_percent_interval;
    return [evaluationMethodLabel(row.first), evaluationMethodLabel(row.second), evaluationCategoryLabel(row.category), pp(row.difference), Array.isArray(interval) && interval.length === 2 ? pp(interval[0]) + " to " + pp(interval[1]) : "Unavailable", metric(row.unit_count), metric(row.resamples)];
  }))); return details;
}
async function evaluation() {
  const content = $("evaluation-content");
  try {
    const report = await api("api/evaluation"); content.replaceChildren();
    const aggregate = report.aggregate;
    if (!aggregate || !aggregate.scores) {
      $("evaluation-status").textContent = report.notice || "No measured evaluation report is available.";
      content.hidden = true; $("evaluation-download").hidden = true; return;
    }
    $("evaluation-status").textContent = `Execution: ${report.execution_status || "Unavailable"} · Completed edit units: ${metric(report.completed_edit_units)} · Unexecuted edit units: ${metric(report.unexecuted_edit_units)}. Abstentions and errors remain in the reported denominators.`;
    content.append(evaluationOverview(report), evaluationAccuracy(aggregate.scores));
    const routing = evaluationRouting(aggregate.rag_retrieval); if (routing) content.append(routing);
    content.append(evaluationHeading("All generated-answer results", "Every method and question category is retained. Accuracy uses the full assigned denominator; query timings cover only instances with a recorded duration."));
    const rows = [];
    for (const [method, categories] of Object.entries(aggregate.scores)) {
      for (const [category, score] of Object.entries(categories)) {
        const latency = aggregate.query_latency?.[method]?.[category] || {};
        rows.push([evaluationMethodLabel(method), evaluationCategoryLabel(category), fraction(score.correct, score.denominator), percentage(score.accuracy), metric(score.abstentions), metric(score.errors_or_unexecuted), metric(latency.p50_seconds, 3), metric(latency.p95_seconds, 3), metric(latency.sample_count)]);
      }
    }
    content.append(table(["Method", "Question type", "Correct / total", "Accuracy", "Abstentions", "Errors / unexecuted", "Query p50 (s)", "Query p95 (s)", "Timed n"], rows));
    const localityRows = [];
    for (const [method, categories] of Object.entries(aggregate.locality_preservation || {})) {
      for (const [category, score] of Object.entries(categories)) {
        localityRows.push([evaluationMethodLabel(method), evaluationCategoryLabel(category), fraction(score.baseline_correct_answer_preserved, score.baseline_correct_denominator), fraction(score.baseline_correct_factual_retained, score.baseline_correct_denominator), fraction(score.unconditional_answer_preserved, score.unconditional_denominator)]);
      }
    }
    if (localityRows.length) {
      const title = document.createElement("h3"); title.textContent = "Preservation and factual correctness"; content.append(title);
      const note = document.createElement("p"); note.className = "muted"; note.textContent = "Verbatim answer preservation differs from factual correctness. Conditional columns include only baseline-correct answers; the unconditional column also includes initially incorrect answers."; content.append(note);
      content.append(table(["Method", "Locality", "Preserved / baseline correct", "Factually retained / baseline correct", "Preserved / all"], localityRows));
    }
    const uncertainty = evaluationUncertainty(aggregate.paired_differences); if (uncertainty) content.append(uncertainty);
    content.append(evaluationResources(report), evaluationHeading("Scope and interpretation"));
    const limitations = document.createElement("p"); limitations.className = "muted"; limitations.textContent = report.limitations || "No additional limitations text was supplied by the local report."; content.append(limitations);
    if (aggregate.latency_protocol) { const protocol = document.createElement("p"); protocol.className = "muted"; protocol.textContent = aggregate.latency_protocol; content.append(protocol); }
    content.hidden = false; $("evaluation-download").hidden = false;
  } catch (error) {
    $("evaluation-status").textContent = "Evaluation unavailable: " + error.message;
    content.hidden = true; $("evaluation-download").hidden = true;
  }
}
async function history() {
  const result = await api("api/history"), container = $("history"); container.replaceChildren();
  if (!result.edits.length) { container.textContent = "No edits in this workspace."; container.classList.add("empty"); return; }
  container.classList.remove("empty");
  for (const row of result.edits) {
    const item = document.createElement("div"); item.className = "history-row";
    const state = document.createElement("span"); state.className = "state"; state.textContent = row.state.replaceAll("_", " ");
    const content = document.createElement("div"), question = document.createElement("p"), target = document.createElement("p"), id = document.createElement("code");
    const method = document.createElement("p"); method.className = "muted"; method.textContent = row.method_label || "Published MEND editor (adapted labels)";
    question.textContent = row.question; target.textContent = "Desired answer: " + row.target; id.textContent = "EDIT " + row.id;
    content.append(question, target, method, id);
    const time = document.createElement("time"); time.className = "muted"; time.textContent = new Date(row.created_at_unix * 1000).toLocaleString();
    const details = document.createElement("details"), summary = document.createElement("summary"), data = document.createElement("pre");
    summary.textContent = "Inspect provenance, patch hashes and model output"; data.textContent = JSON.stringify(row, null, 2); details.append(summary, data);
    const exports = document.createElement("p"), recordLink = document.createElement("a");
    recordLink.href = "api/export/" + row.id; recordLink.textContent = "Download evidence JSON"; exports.append(recordLink);
    if (row.patch) { const patchLink = document.createElement("a"); patchLink.href = "api/patch/" + row.id; patchLink.textContent = "Download parameter patch (large)"; exports.append(document.createTextNode(" · "), patchLink); }
    if (["candidate", "rolled_back", "inactive_after_restart"].includes(row.state) && row.record_format === "editguard-local-candidate-v1") {
      const approve = document.createElement("button"); approve.className = "button secondary compact"; approve.dataset.activateCandidate = row.id; approve.textContent = "Approve and activate this candidate";
      approve.addEventListener("click", () => activateCandidate(row.id)); details.append(approve);
    }
    details.append(exports); item.append(state, content, time, details); container.append(item);
  }
}
$("generate").addEventListener("click", () => perform("generate"));
$("apply").addEventListener("click", () => perform("edit"));
$("rollback").addEventListener("click", () => perform("rollback"));
$("activate").addEventListener("click", () => activateCandidate(candidate));
$("refresh-evaluation").addEventListener("click", evaluation);
$("fixture").addEventListener("change", selectFixture);
$("question").addEventListener("input", resetAnswers);
$("target").addEventListener("input", resetAnswers);
$("method").addEventListener("change", () => {
  $("method-note").textContent = $("method").value === "mend" ? "Uses the published pretrained learned gradient transform with context-aligned labels." : "Adapted baseline: 15 AdamW steps at learning rate 0.0002 on layer 31 fc1/fc2 only. Uses FP32 master optimizer weights to reduce memory use.";
  resetAnswers();
});
$("upload").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 5 * 1024 * 1024 || !["image/jpeg","image/png","image/webp"].includes(file.type)) { event.target.value = ""; return message("Use a JPEG, PNG or WebP image up to 5 MiB."); }
  const reader = new FileReader(); reader.onload = () => { upload = reader.result.split(",")[1]; if (objectURL) URL.revokeObjectURL(objectURL); objectURL = URL.createObjectURL(file); $("preview").src = objectURL; $("image-source").textContent = "Your uploaded image · validity and dimensions checked by the server · not an evaluation case."; resetAnswers(); message(""); }; reader.readAsDataURL(file);
});
(async () => {
  try {
    const result = await api("api/fixtures"); fixtures = result.cases;
    for (const row of fixtures) { const option = document.createElement("option"); option.value = row.id; option.textContent = row.id.replaceAll("_", " "); $("fixture").append(option); }
    selectFixture(); await history(); await status(); await evaluation();
  } catch (error) { message(error.message); }
  setInterval(status, 3000);
})();
