"use strict";
const $ = id => document.getElementById(id);
let csrf = "", busy = false, remoteBusy = false, ready = false, active = null, upload = null, objectURL = null, fixtures = [];
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
  $("fixture").disabled = busy; $("upload").disabled = busy;
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
  busy = true; controls(); message(""); $("operation-meta").textContent = operation === "edit" ? "Generating a baseline, applying MEND, saving the patch, then generating again…" : operation === "rollback" ? "Restoring saved original tensors and checking their exact hashes…" : "Generating from the current model weights…";
  try {
    const result = await api("api/" + operation, {question:$("question").value.trim(), target:$("target").value.trim(), ...imagePayload()});
    if (operation === "generate") {
      text(active ? "after-answer" : "before-answer", result.answer);
      const other = $(active ? "before-answer" : "after-answer");
      other.textContent = active ? "Original model answer was not generated for this query." : "Apply a correction to compare.";
      other.classList.add("empty");
      $("operation-meta").textContent = `Generated in ${result.seconds.toFixed(2)} s · ${active ? "Active edited model" : "Original model"} · Unverified output`;
    } else if (operation === "edit") {
      text("before-answer", result.edit.before_answer); text("after-answer", result.edit.after_answer);
      $("operation-meta").textContent = `Patch saved · ${result.edit.seconds.toFixed(2)} s for the complete edit workflow · Generated answers, not factual verification`;
    } else {
      $("operation-meta").textContent = "Original editable tensors restored exactly; hashes match. Generate again to inspect restored behavior.";
      $("after-answer").textContent = "Edit rolled back."; $("after-answer").classList.add("empty");
    }
    await history();
  } catch(error) { message(error.message); await history(); }
  finally { busy = false; await status(); controls(); }
}
function selectFixture() {
  upload = null; $("upload").value = "";
  if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null;
  const item = fixtures.find(row => row.id === $("fixture").value);
  if (item) { $("preview").src = item.image_url; $("question").value = item.question; }
  $("image-source").textContent = "Author-provided training fixture · not a held-out evaluation.";
}
async function history() {
  const result = await api("api/history"), container = $("history"); container.replaceChildren();
  if (!result.edits.length) { container.textContent = "No edits in this workspace."; container.classList.add("empty"); return; }
  container.classList.remove("empty");
  for (const row of result.edits) {
    const item = document.createElement("div"); item.className = "history-row";
    const state = document.createElement("span"); state.className = "state"; state.textContent = row.state.replaceAll("_", " ");
    const content = document.createElement("div"), question = document.createElement("p"), target = document.createElement("p"), id = document.createElement("code");
    question.textContent = row.question; target.textContent = "Desired answer: " + row.target; id.textContent = "EDIT " + row.id;
    content.append(question, target, id);
    const time = document.createElement("time"); time.className = "muted"; time.textContent = new Date(row.created_at_unix * 1000).toLocaleString();
    const details = document.createElement("details"), summary = document.createElement("summary"), data = document.createElement("pre");
    summary.textContent = "Inspect provenance, patch hashes and model output"; data.textContent = JSON.stringify(row, null, 2); details.append(summary, data);
    const exports = document.createElement("p"), recordLink = document.createElement("a");
    recordLink.href = "api/export/" + row.id; recordLink.textContent = "Download evidence JSON"; exports.append(recordLink);
    if (row.patch) { const patchLink = document.createElement("a"); patchLink.href = "api/patch/" + row.id; patchLink.textContent = "Download parameter patch (large)"; exports.append(document.createTextNode(" · "), patchLink); }
    details.append(exports); item.append(state, content, time, details); container.append(item);
  }
}
$("generate").addEventListener("click", () => perform("generate"));
$("apply").addEventListener("click", () => perform("edit"));
$("rollback").addEventListener("click", () => perform("rollback"));
$("fixture").addEventListener("change", selectFixture);
$("upload").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 5 * 1024 * 1024 || !["image/jpeg","image/png","image/webp"].includes(file.type)) { event.target.value = ""; return message("Use a JPEG, PNG or WebP image up to 5 MiB."); }
  const reader = new FileReader(); reader.onload = () => { upload = reader.result.split(",")[1]; if (objectURL) URL.revokeObjectURL(objectURL); objectURL = URL.createObjectURL(file); $("preview").src = objectURL; $("image-source").textContent = "Your uploaded image · validity and dimensions checked by the server · not an evaluation case."; message(""); }; reader.readAsDataURL(file);
});
(async () => {
  try {
    const result = await api("api/fixtures"); fixtures = result.cases;
    for (const row of fixtures) { const option = document.createElement("option"); option.value = row.id; option.textContent = row.id.replaceAll("_", " "); $("fixture").append(option); }
    selectFixture(); await history(); await status();
  } catch (error) { message(error.message); }
  setInterval(status, 3000);
})();
