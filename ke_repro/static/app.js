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
  const wrapper = document.createElement("div"); wrapper.className = "table-scroll";
  const result = document.createElement("table"), head = document.createElement("thead"), header = document.createElement("tr"), body = document.createElement("tbody");
  for (const label of headers) { const cell = document.createElement("th"); cell.textContent = label; header.append(cell); }
  head.append(header);
  for (const values of rows) { const row = document.createElement("tr"); for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); } body.append(row); }
  result.append(head, body); wrapper.append(result); return wrapper;
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
    const methodLabel = key => ({base:"Unchanged base",prompt_only:"Prompt-only correction",updated_rag:"Updated image + text RAG",mend_context:"MEND (context aligned)",ft_last_layer:"Last-layer fine-tuning"})[key] || key;
    const categoryLabel = key => key.replaceAll("_", " ");
    const rows = [];
    for (const [method, categories] of Object.entries(aggregate.scores)) {
      for (const [category, score] of Object.entries(categories)) {
        const latency = aggregate.query_latency?.[method]?.[category] || {};
        rows.push([methodLabel(method), categoryLabel(category), fraction(score.correct, score.denominator), typeof score.accuracy === "number" ? metric(score.accuracy * 100, 1) + "%" : "Unavailable", metric(score.abstentions), metric(score.errors_or_unexecuted), metric(latency.p50_seconds, 3), metric(latency.p95_seconds, 3), metric(latency.sample_count)]);
      }
    }
    content.append(table(["Method", "Question type", "Correct / total", "Accuracy", "Abstentions", "Errors / unexecuted", "Query p50 (s)", "Query p95 (s)", "Timed n"], rows));
    const localityRows = [];
    for (const [method, categories] of Object.entries(aggregate.locality_preservation || {})) {
      for (const [category, score] of Object.entries(categories)) {
        localityRows.push([methodLabel(method), categoryLabel(category), fraction(score.baseline_correct_answer_preserved, score.baseline_correct_denominator), fraction(score.baseline_correct_factual_retained, score.baseline_correct_denominator), fraction(score.unconditional_answer_preserved, score.unconditional_denominator)]);
      }
    }
    if (localityRows.length) {
      const title = document.createElement("h3"); title.textContent = "Preservation and factual correctness"; content.append(title);
      const note = document.createElement("p"); note.className = "muted"; note.textContent = "Verbatim answer preservation differs from factual correctness. Conditional columns include only baseline-correct answers; the unconditional column also includes initially incorrect answers."; content.append(note);
      content.append(table(["Method", "Locality", "Preserved / baseline correct", "Factually retained / baseline correct", "Preserved / all"], localityRows));
    }
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
