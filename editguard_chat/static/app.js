"use strict";

const $ = id => document.getElementById(id);
const CATEGORY_LABELS = {direct: "Direct correction", direct_paraphrase: "Direct paraphrase", two_hop: "Two-hop question", two_hop_paraphrase: "Two-hop paraphrase", same_entity_other_relation: "Same service, another property", similar_entity_same_relation: "Similarly named service", wrong_version_or_tier: "Older release", unrelated: "Unrelated fact"};
const state = {token: "", status: null, chatId: null, chats: [], messages: [], corrections: [], revision: 0, busy: false, attachment: null, resultsLoaded: false};
let toastTimer;

class ApiError extends Error {
  constructor(message, status, body) { super(message); this.status = status; this.body = body; }
}

async function api(path, body) {
  const options = {cache: "no-store", headers: {}};
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.headers["X-EditGuard-Token"] = state.token;
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let result;
  try { result = await response.json(); } catch { throw new ApiError("The server returned an unreadable response. Refresh the page or inspect its launch log.", response.status, {}); }
  if (!response.ok) {
    const message = typeof result.error === "string" ? result.error : typeof result.detail === "string" ? result.detail : result.error?.message || "The operation could not be completed.";
    throw new ApiError(message, response.status, result);
  }
  return result;
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function clearError() { $("global-error").hidden = true; $("global-error").textContent = ""; }
function showError(error) { $("global-error").textContent = error.message || String(error); $("global-error").hidden = false; }
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6000);
}

function dateLabel(value) {
  if (!value) return "";
  const date = new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, {month:"short", day:"numeric", hour:"numeric", minute:"2-digit"});
}

function setRevision(revision) {
  if (revision !== undefined && revision !== null) state.revision = revision;
  $("revision-pill").textContent = "Knowledge r" + state.revision;
}

function modelReady() { return state.status?.model?.ready === true; }

function renderStatus(status) {
  state.status = status;
  state.token = status.token || status.csrf_token || state.token;
  setRevision(status.revision);
  const model = status.model || {};
  const ready = model.ready === true;
  const failed = !!model.error;
  $("sidebar-model").textContent = model.name || "Local conversation model";
  $("sidebar-model-detail").textContent = failed ? "Model unavailable · see launch log" : ready ? (model.backend || "Model") + " · ready to chat" : "Waiting for the model";
  if (!$("compare-dialog").open) $("comparison-model-label").textContent = status.comparison_label ? "WITHOUT CORRECTION CONTEXT" : "ORIGINAL MODEL";
  $("knowledge-context-description").textContent = status.retrieval?.method === "all_approved_facts_learned_applicability" ? "All active approved facts are supplied to the trained reasoner, up to 20 records. New corrections update external memory; model weights remain unchanged." : "Corrections are supplied to the model when retrieved. They do not rewrite the base model’s weights.";
  if (status.retrieval?.method === "two_stage_lexical_approved_memory") $("knowledge-context-description").textContent = "Up to four approved facts are selected from a library of at most 200 using two-stage question-only retrieval. New corrections update external memory; model weights remain unchanged.";
  $("model-status").textContent = ready ? "Model ready" : failed ? "Model unavailable" : "Model loading";
  document.querySelectorAll(".model-pill .status-dot, #sidebar-status-dot").forEach(dot => { dot.className = "status-dot " + (ready ? "ready" : failed ? "failed" : ""); });
  const images = model.supports_images === true;
  $("attach-button").classList.toggle("unavailable", !images);
  $("attach-button").title = images ? "Attach an optional JPEG, PNG or WebP" : "Image input is unavailable with the current model";
  $("image-input").disabled = !images || state.busy;
  $("send-button").disabled = !ready || state.busy;
  $("composer-model").textContent = ready ? (model.name || "Local model") + " · Generated answers can be incorrect." : failed ? "Model unavailable. " + model.error : "Waiting for the local model to become ready…";
}

async function refreshStatus() {
  try { renderStatus(await api("/api/status")); }
  catch (error) {
    $("model-status").textContent = "Server unavailable";
    $("sidebar-model-detail").textContent = "Check the local server is running";
    $("send-button").disabled = true;
  }
}

function showView(name) {
  const view = ["chat", "knowledge", "results"].includes(name) ? name : "chat";
  document.querySelectorAll(".view").forEach(node => { node.hidden = node.id !== "view-" + view; });
  document.querySelectorAll(".nav-item").forEach(node => { node.classList.toggle("active", node.dataset.view === view); });
  $("view-label").textContent = {chat:"Chat", knowledge:"Knowledge", results:"Research results"}[view];
  if (location.hash !== "#" + view) history.replaceState(null, "", "#" + view);
  if (view === "knowledge") loadCorrections().catch(showError);
  if (view === "results" && !state.resultsLoaded) loadResults().catch(showError);
  if (view === "chat") $("chat-input").focus();
}

function setBusy(busy, label = "Generating an answer…") {
  state.busy = busy;
  $("thinking").hidden = !busy;
  $("thinking").lastElementChild.textContent = label;
  $("send-button").disabled = busy || !modelReady();
  $("image-input").disabled = busy || state.status?.model?.supports_images !== true;
  $("chat-input").disabled = busy;
  $("new-chat").disabled = busy;
  $("mobile-new-chat").disabled = busy;
  $("mobile-chat-select").disabled = busy || !state.chats.length;
  document.querySelectorAll(".message-actions button").forEach(button => { button.disabled = busy; });
  if (busy) scrollToEnd();
}

function scrollToEnd() { requestAnimationFrame(() => { $("message-scroller").scrollTop = $("message-scroller").scrollHeight; }); }

function renderChats() {
  const list = $("chat-list");
  list.replaceChildren();
  const mobileSelect = $("mobile-chat-select"); mobileSelect.replaceChildren();
  const placeholder = element("option", "", state.chats.length ? "Select a conversation" : "No conversations yet"); placeholder.value = ""; mobileSelect.append(placeholder);
  for (const chat of state.chats) { const option = element("option", "", chat.title || "New conversation"); option.value = chat.id; mobileSelect.append(option); }
  mobileSelect.value = state.chatId || "";
  mobileSelect.disabled = state.busy || !state.chats.length;
  if (!state.chats.length) { list.append(element("p", "sidebar-empty", "Your conversations will appear here.")); return; }
  for (const chat of state.chats) {
    const button = element("button", "chat-link" + (String(chat.id) === String(state.chatId) ? " selected" : ""), chat.title || "New conversation");
    button.title = chat.title || "New conversation";
    button.addEventListener("click", () => { if (!state.busy) openChat(chat.id).catch(showError); });
    list.append(button);
  }
}

async function loadChats() {
  const result = await api("/api/chats");
  state.chats = result.chats || [];
  renderChats();
}

async function openChat(id) {
  clearError();
  const data = await api("/api/chats/" + encodeURIComponent(id));
  state.chatId = data.chat?.id ?? id;
  state.messages = data.messages || data.chat?.messages || [];
  localStorage.setItem("editguard-chat-id", String(state.chatId));
  $("conversation-title").textContent = data.chat?.title || "A conversation can change.";
  $("revision-notice").hidden = true;
  renderChats(); renderMessages(); showView("chat");
}

async function createChat() {
  const result = await api("/api/chats", {});
  state.chatId = result.chat?.id ?? result.id;
  if (state.chatId == null) throw new Error("The server did not return a conversation identifier.");
  state.messages = [];
  localStorage.setItem("editguard-chat-id", String(state.chatId));
  $("conversation-title").textContent = "A conversation can change.";
  $("revision-notice").hidden = true;
  renderMessages(); await loadChats(); showView("chat");
}

function evidenceRows(evidence) {
  if (Array.isArray(evidence)) return evidence;
  if (!evidence || typeof evidence !== "object") return [];
  return evidence.corrections || evidence.matches || evidence.retrieved || evidence.sources || [];
}

function evidenceNode(evidence) {
  const wrap = element("div", "evidence-list");
  for (const match of evidenceRows(evidence)) {
    if (!match || typeof match !== "object") continue;
    const chip = element("div", "evidence-chip");
    chip.append(element("strong", "", match.question || match.title || "Approved correction"));
    if (match.answer || match.content) chip.append(element("span", "", match.answer || match.content));
    if (match.source) chip.append(element("span", "evidence-source", "Source: " + match.source));
    const details = [match.scope ? "Scope: " + match.scope : "", typeof match.score === "number" ? "Retrieval similarity: " + match.score.toFixed(3) + " (not a probability)" : ""].filter(Boolean);
    if (details.length) chip.append(element("span", "evidence-source", details.join(" · ")));
    wrap.append(chip);
  }
  if (wrap.childElementCount) wrap.append(element("p", "routing-note", evidence?.notice || "Supplied correction sources support inspection; they do not verify every sentence generated by the model."));
  else if (evidence?.reason || evidence?.notice) {
    const reasons = {original_model_requested:"Answered without approved corrections.", no_active_corrections:"No active corrections were supplied to this response.", explicit_scope_mismatch:"The available corrections have a different scope from this question.", below_development_threshold:"No correction met the current retrieval threshold.", ambiguous_corrections:"Competing corrections need a scope clarification. This clarification is produced by a deterministic rule."};
    if (evidence.notice || reasons[evidence.reason]) wrap.append(element("p", "routing-note", evidence.notice || reasons[evidence.reason]));
  }
  return wrap;
}

function previousUser(index) { for (let i = index - 1; i >= 0; i--) if (state.messages[i].role === "user") return state.messages[i]; return null; }

function actionButton(text, className, action) {
  const button = element("button", className, text);
  button.addEventListener("click", action);
  button.disabled = state.busy;
  return button;
}

function renderMessages() {
  $("welcome").hidden = state.messages.length > 0;
  const list = $("messages"); list.replaceChildren();
  state.messages.forEach((message, index) => {
    const user = message.role === "user";
    const article = element("article", "message " + (user ? "user" : "assistant"));
    if (!user) article.append(element("span", "assistant-avatar", "eg"));
    const body = element("div", "message-body");
    if (!user) {
      const heading = element("div", "message-heading");
      heading.append(element("strong", "", "EditGuard"));
      const used = evidenceRows(message.evidence).length > 0;
      const clarification = message.evidence?.generation_kind === "deterministic_clarification";
      heading.append(element("span", "message-mode", clarification ? "Scope clarification · rule-based" : message.mode === "original" ? (message.evidence?.method === "without_correction_context" ? "Without correction context" : "Original model") : used ? "Correction context supplied" : "Model response"));
      if (message.knowledge_revision != null) heading.append(element("span", "message-mode", "r" + message.knowledge_revision));
      body.append(heading);
    }
    if (message.image_url) { const image = element("img", "message-image"); image.src = message.image_url; image.alt = "Image attached to this message"; body.append(image); }
    body.append(element("div", "message-text", message.content || "[No answer generated]"));
    if (!user) body.append(evidenceNode(message.evidence));
    const date = dateLabel(message.created_at);
    if (date) body.append(element("div", "message-time", date));
    if (!user) {
      const question = previousUser(index);
      const actions = element("div", "message-actions");
      if (question) {
        actions.append(actionButton("⇄ Compare original & corrected", "compare-action", () => compare(question).catch(showError)));
        actions.append(actionButton("✎ Correct this answer", "", () => draftCorrection(question.content)));
        actions.append(actionButton("↻ Ask again", "", () => { $("chat-input").value = question.content || ""; resizeComposer(); showView("chat"); }));
      }
      body.append(actions);
    }
    article.append(body); list.append(article);
  });
  scrollToEnd();
}

function resizeComposer() {
  const input = $("chat-input"); input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 150) + "px";
}

function clearAttachment() {
  if (state.attachment?.preview) URL.revokeObjectURL(state.attachment.preview);
  state.attachment = null;
  $("image-input").value = "";
  $("attachment-preview").hidden = true;
  $("attachment-image").removeAttribute("src");
}

async function attachImage(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error("Choose a JPEG, PNG or WebP image.");
  if (file.size > 5 * 1024 * 1024) throw new Error("The image must be no larger than 5 MiB.");
  const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error("The image could not be read.")); reader.readAsDataURL(file); });
  clearAttachment();
  state.attachment = {base64: String(data).split(",")[1], mime: file.type, preview: URL.createObjectURL(file), name: file.name};
  $("attachment-image").src = state.attachment.preview;
  $("attachment-name").textContent = file.name;
  $("attachment-preview").hidden = false;
}

async function sendMessage(event) {
  event.preventDefault(); clearError();
  const content = $("chat-input").value.trim();
  if (!content || state.busy) return;
  if (!modelReady()) { showError(new Error("Wait for the local model to be ready before sending a message.")); return; }
  setBusy(true);
  try {
    if (!state.chatId) await createChat();
    const payload = {chat_id: state.chatId, message: content, mode: $("use-knowledge").checked ? "edited" : "original"};
    if (state.attachment) { payload.image_base64 = state.attachment.base64; payload.image_mime = state.attachment.mime; }
    $("welcome").hidden = true;
    const pending = element("article", "message user");
    const pendingBody = element("div", "message-body");
    if (state.attachment) { const image = element("img", "message-image"); image.src = state.attachment.preview; image.alt = "Image attached to the message being generated"; pendingBody.append(image); }
    pendingBody.append(element("div", "message-text", content)); pending.append(pendingBody); $("messages").append(pending); scrollToEnd();
    const result = await api("/api/chat", payload);
    setRevision(result.revision);
    $("chat-input").value = ""; resizeComposer(); clearAttachment();
    $("revision-notice").hidden = true;
    if (result.user && result.assistant) { state.messages.push(result.user, result.assistant); renderMessages(); }
    else await openChat(state.chatId);
    await loadChats();
    const currentChat = state.chats.find(chat => String(chat.id) === String(state.chatId));
    if (currentChat?.title) $("conversation-title").textContent = currentChat.title;
  } catch (error) { renderMessages(); throw error; }
  finally { setBusy(false); $("chat-input").focus(); }
}

async function compare(question) {
  if (state.busy) return;
  state.compareQuestion = question;
  clearError();
  const historical = $("compare-baseline-mode").value === "revision";
  const baseline = Number($("compare-baseline-revision").value);
  if (historical && (!Number.isInteger(baseline) || baseline < 0 || baseline > state.revision)) throw new Error("Choose an existing integer knowledge revision.");
  $("compare-baseline-revision").max = state.revision;
  $("comparison-model-label").textContent = historical ? "APPROVED MEMORY r" + baseline : "WITHOUT CORRECTION CONTEXT";
  $("compare-question").textContent = question.content;
  $("compare-original").textContent = historical ? "Generating with approved memory at revision " + baseline + "…" : "Generating without correction context…";
  $("compare-edited").textContent = "Generating with current approved knowledge…";
  $("compare-evidence").replaceChildren();
  $("compare-revision").textContent = "Knowledge r" + state.revision;
  if (!$("compare-dialog").open) $("compare-dialog").showModal();
  setBusy(true, "Generating the comparison…");
  try {
    const payload = {chat_id: state.chatId, message_id: question.id};
    if (historical) payload.baseline_revision = baseline;
    const result = await api("/api/compare", payload);
    $("comparison-model-label").textContent = result.baseline_label || "WITHOUT CORRECTION CONTEXT";
    $("compare-original").textContent = result.original?.content || result.original?.answer || "[No answer generated]";
    $("compare-edited").textContent = result.edited?.content || result.edited?.answer || "[No answer generated]";
    $("compare-revision").textContent = "Knowledge r" + (result.revision ?? state.revision);
    $("compare-evidence").append(evidenceNode(result.edited?.evidence));
  } catch (error) {
    $("compare-original").textContent = "Comparison could not be generated.";
    $("compare-edited").textContent = error.message;
    throw error;
  } finally { setBusy(false); }
}

function draftCorrection(question) {
  $("correction-question").value = question || "";
  $("correction-answer").value = "";
  ["fact-entity", "fact-relation", "fact-scope", "fact-aliases"].forEach(id => { $(id).value = ""; });
  resetReplacement(); showView("knowledge"); $("correction-answer").focus();
}

async function loadCorrections() {
  const result = await api("/api/corrections");
  state.corrections = result.corrections || [];
  state.events = result.events || [];
  setRevision(result.revision); renderCorrections();
  renderAuditHistory();
  await refreshExampleHelper();
}

function isActive(row) { return row.status === "active" || row.state === "active"; }

function renderCorrections() {
  const active = state.corrections.filter(isActive).length;
  $("active-count").textContent = active;
  $("total-count").textContent = state.corrections.length;
  $("nav-knowledge-count").textContent = active;
  const list = $("correction-list"); list.replaceChildren();
  const query = $("correction-search").value.trim().toLocaleLowerCase();
  const rows = state.corrections.filter(row => ($("show-withdrawn").checked || isActive(row)) &&
    [row.question, row.answer, row.scope, row.source, row.identity?.entity_id, row.identity?.relation_id, row.identity?.scope_id].some(value => String(value || "").toLocaleLowerCase().includes(query)));
  if (!rows.length) {
    const empty = element("div", "empty-library");
    empty.append(element("span", "", "▤"), element("h3", "", active ? "No matching corrections." : "A clean slate."), element("p", "", "Publish a sourced correction to change what your assistant uses."));
    list.append(empty); return;
  }
  for (const row of rows) {
    const activeRow = isActive(row);
    const card = element("article", "correction-card" + (activeRow ? "" : " inactive"));
    const top = element("div", "correction-card-top");
    top.append(element("span", "correction-status", activeRow ? "ACTIVE" : (row.status || row.state || "inactive").replaceAll("_", " ").toUpperCase()), element("span", "version-tag", "VERSION " + (row.version || 1)));
    card.append(top, element("h3", "", row.question), element("p", "correction-answer", row.answer));
    const source = element("p", "correction-source"); source.append(element("span", "", "SOURCE  "), document.createTextNode(row.source || "Unspecified")); card.append(source);
    if (row.scope) card.append(element("span", "correction-scope", row.scope));
    if (row.identity?.entity_id) card.append(element("p", "routing-note", "Canonical fact: " + row.identity.entity_id + " / " + row.identity.relation_id + " / " + row.identity.scope_id));
    if (row.supersedes) card.append(element("p", "routing-note", "Replaces an earlier version · " + String(row.supersedes).slice(0, 12)));
    const bottom = element("div", "correction-card-bottom");
    bottom.append(element("time", "", dateLabel(row.created_at)));
    const actions = element("div", "message-actions"); actions.style.marginTop = "0";
    if (activeRow) actions.append(actionButton("Publish a new version", "", () => startReplacement(row)));
    if (activeRow && row.supersedes) actions.append(actionButton("Roll back to previous version", "", () => changeCorrection(row.id, "rollback").catch(showError)));
    actions.append(actionButton(activeRow ? "Withdraw" : "Restore version", "", () => changeCorrection(row.id, activeRow ? "withdraw" : "restore").catch(showError)));
    bottom.append(actions); card.append(bottom); list.append(card);
  }
}

function renderAuditHistory() {
  const target = $("audit-history"); target.replaceChildren();
  const events = [...(state.events || [])].reverse();
  if (!events.length) { target.append(element("p", "muted", "Approved changes will appear here.")); return; }
  for (const event of events) {
    const fact = state.corrections.find(row => row.id === event.correction_id);
    const row = element("div", "audit-entry");
    row.append(element("strong", "", "r" + event.revision + " · " + event.action), element("span", "", fact?.question || event.correction_id),
      element("small", "", dateLabel(event.created_at) + " · " + (fact ? "v" + fact.version + " · " + fact.answer : event.actor)));
    target.append(row);
  }
}

function resetReplacement() {
  $("supersedes").value = ""; $("correction-conflict").hidden = true; $("correction-conflict").replaceChildren();
  $("cancel-replacement").hidden = true; $("publish-button").firstChild.textContent = "Approve & publish ";
}

function startReplacement(row) {
  $("correction-question").value = row.question || "";
  $("correction-answer").value = row.answer || "";
  $("correction-source").value = row.source || "";
  $("correction-scope").value = row.scope || "";
  $("fact-entity").value = row.identity?.entity_id || "";
  $("fact-relation").value = row.identity?.relation_id || "";
  $("fact-scope").value = row.identity?.scope_id || "";
  $("fact-aliases").value = (row.identity?.aliases || []).join(", ");
  $("supersedes").value = row.id;
  $("cancel-replacement").hidden = false;
  $("publish-button").firstChild.textContent = "Approve & replace version ";
  $("correction-conflict").hidden = true;
  $("correction-answer").focus();
}

function showConflict(error) {
  const wrap = $("correction-conflict"); wrap.replaceChildren(); wrap.hidden = false;
  wrap.append(element("p", "", error.message));
  const conflicts = error.body?.conflicts || error.body?.detail?.conflicts || error.body?.error?.conflicts || [];
  for (const conflict of conflicts) {
    const row = typeof conflict === "string" ? state.corrections.find(item => item.id === conflict) : conflict;
    if (!row?.id) continue;
    const button = element("button", "", "Replace: " + (row.question || String(row.id)));
    button.type = "button";
    button.addEventListener("click", () => { $("supersedes").value = row.id; $("cancel-replacement").hidden = false; $("publish-button").firstChild.textContent = "Approve & replace version "; wrap.replaceChildren(element("p", "", "The selected existing version will be superseded when you explicitly approve and publish.")); });
    wrap.append(button);
  }
}

async function publishCorrection(event) {
  event.preventDefault(); clearError();
  const payload = {question: $("correction-question").value.trim(), answer: $("correction-answer").value.trim(), scope: $("correction-scope").value.trim(), source: $("correction-source").value.trim()};
  if ([$("fact-entity"), $("fact-relation"), $("fact-scope")].some(input => input.value.trim())) {
    payload.identity = {entity_id: $("fact-entity").value.trim(), relation_id: $("fact-relation").value.trim(),
      scope_id: $("fact-scope").value.trim(), aliases: $("fact-aliases").value.split(",").map(value => value.trim()).filter(Boolean)};
  }
  if ($("supersedes").value) payload.supersedes = $("supersedes").value;
  $("publish-button").disabled = true;
  try {
    const result = await api("/api/corrections", payload);
    setRevision(result.revision); $("correction-form").reset(); resetReplacement();
    await loadCorrections(); $("revision-notice").hidden = false;
    toast("Correction published. Return to Chat and ask again, or compare an existing answer.");
  } catch (error) { if (error.status === 409) showConflict(error); else throw error; }
  finally { $("publish-button").disabled = false; }
}

async function changeCorrection(id, action) {
  clearError();
  const result = await api("/api/corrections/" + encodeURIComponent(id) + "/" + action, {});
  setRevision(result.revision); await loadCorrections(); $("revision-notice").hidden = false;
  toast(action === "withdraw" ? "Correction withdrawn. New answers will use the current knowledge state." : action === "rollback" ? "Previous version restored in one transaction. Return to Chat to inspect its effect." : "Version restored. Return to Chat to inspect its effect.");
}

function resultCard(value, caption) { const card = element("div", "result-card"); card.append(element("strong", "", value), element("span", "", caption)); return card; }

let exampleEdit = null;
let exampleDefinition = null;
async function refreshExampleHelper() {
  $("approve-example").disabled = state.revision !== 0;
  exampleDefinition ||= await api("/api/examples/india");
  const definition = exampleDefinition.world;
  const active = state.corrections.find(row => row.status === "active" && row.identity?.entity_id === definition.edit.entity_id && row.identity?.relation_id === definition.edit.relation_id && row.identity?.scope_id === definition.edit.scope_id);
  if (active) {
    const updated = definition.current_facts.find(row => row.id === definition.edit.fact_id);
    exampleEdit = {...active, answer: updated.answer, source: updated.source};
    $("draft-example-edit").hidden = active.answer === updated.answer;
    const links = $("example-questions"); links.replaceChildren();
    exampleDefinition.questions.forEach(question => {
      const button = element("button", "", question); button.type = "button";
      button.addEventListener("click", () => { showView("chat"); $("chat-input").value = question; resizeComposer(); }); links.append(button);
    });
    $("example-status").textContent = "Fictional handbook for " + definition.entity + ". The active ownership answer is " + active.answer + ". Compare with an earlier revision to inspect a published change.";
  } else { exampleEdit = null; $("draft-example-edit").hidden = true; $("example-questions").replaceChildren(); $("example-status").textContent = state.revision ? "This workspace has knowledge history. Existing facts stay unchanged. Use a fresh workspace for the India-themed example, or edit the library below." : "No fictional facts are active until you approve the handbook."; }
}
$("approve-example").addEventListener("click", async () => {
  if (state.busy) return;
  $("approve-example").disabled = true;
  try {
    const result = await api("/api/examples/india/approve", {});
    setRevision(result.revision); await loadCorrections();
    exampleEdit = result.suggested_edit;
    $("draft-example-edit").hidden = false;
    $("example-status").textContent = "Approved fictional handbook for " + result.example.entity + " at revision " + result.revision + ". Ask a question first, then explicitly publish the ownership update.";
    const links = $("example-questions"); links.replaceChildren();
    result.example.questions.forEach(question => {
      const button = element("button", "", question); button.type = "button";
      button.addEventListener("click", () => { showView("chat"); $("chat-input").value = question; resizeComposer(); }); links.append(button);
    });
  } catch (error) { showError(error); $("approve-example").disabled = false; }
});
$("draft-example-edit").addEventListener("click", () => {
  if (exampleEdit) { startReplacement(exampleEdit); $("example-status").textContent = "The ownership update is a draft. Review its answer and source, then click Approve & replace version to activate it."; }
});

let workspaceImport = null;
$("workspace-export").addEventListener("click", async () => {
  try {
    const snapshot = await api("/api/workspace/export");
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], {type: "application/json"}));
    const link = element("a"); link.href = url; link.download = "editguard-workspace-r" + snapshot.revision + ".json";
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Workspace exported. Store the file privately; it contains your conversations.");
  } catch (error) { showError(error); }
});
$("workspace-file").addEventListener("change", async event => {
  workspaceImport = null; $("workspace-restore").disabled = true; $("workspace-preview").textContent = "";
  try {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 6 * 1024 * 1024) throw new Error("Choose a workspace JSON file up to 6 MiB.");
    const value = JSON.parse(await file.text());
    if (value.schema !== "editguard-workspace-v1" || !value.tables || !Array.isArray(value.tables.chats) || !Array.isArray(value.tables.corrections)) throw new Error("Choose an EditGuard workspace snapshot.");
    workspaceImport = value;
    $("workspace-preview").textContent = value.tables.chats.length + " conversations, " + value.tables.corrections.length + " fact versions; revision " + value.revision + ". The server checks integrity and all references before import.";
    $("workspace-restore").disabled = false;
  } catch (error) { showError(error); }
});
$("workspace-restore").addEventListener("click", async () => {
  if (!workspaceImport || state.busy) return;
  $("workspace-restore").disabled = true;
  try {
    const result = await api("/api/workspace/restore", {snapshot: workspaceImport});
    workspaceImport = null; $("workspace-file").value = ""; setRevision(result.revision);
    await Promise.all([loadChats(), loadCorrections()]);
    $("workspace-preview").textContent = "Workspace restored successfully at revision " + result.revision + ".";
    toast("Workspace restored. Choose a saved conversation or start a new one.");
  } catch (error) { showError(error); $("workspace-restore").disabled = false; }
});

function evidenceTable(headings, rows) {
  const wrap = element("div", "result-table-wrap"), table = element("table"), head = element("thead"), tr = element("tr");
  headings.forEach(text => tr.append(element("th", "", text))); head.append(tr); table.append(head);
  const body = element("tbody");
  rows.forEach(values => { const row = element("tr"); values.forEach(text => row.append(element("td", "", String(text ?? "Unavailable")))); body.append(row); });
  table.append(body); wrap.append(table); return wrap;
}

async function addAnswerExplorer(target, report) {
  const panel = element("section", "result-panel answer-explorer");
  panel.append(element("h2", "", "Inspect the saved answers"), element("p", "routing-note", "Read actual model outputs and supplied facts. These are saved experiment records, not fresh chat responses. Filtering does not change reported denominators."));
  const filters = element("div", "case-filters"), controls = {};
  const choices = {condition: Object.keys(report.conditions), outcome: ["wrong", "abstain", "execution_error", "unexecuted", "correct"], category: Object.keys(report.conditions["R-RAG"].by_category)};
  for (const [name, values] of Object.entries(choices)) {
    const label = element("label", "", name[0].toUpperCase() + name.slice(1));
    const select = element("select"); select.setAttribute("aria-label", "Saved answer " + name);
    const all = element("option", "", "All " + (name === "category" ? "categories" : name + "s")); all.value = ""; select.append(all);
    values.forEach(value => { const option = element("option", "", name === "category" ? CATEGORY_LABELS[value] || value : value.replaceAll("_", " ")); option.value = value; select.append(option); });
    controls[name] = select; label.append(select); filters.append(label);
  }
  const load = element("button", "quiet-button", "Show saved answers"); load.type = "button"; filters.append(load);
  const summary = element("p", "routing-note"), list = element("div", "case-list");
  const more = element("button", "quiet-button", "Load next 20 saved answers"); more.type = "button"; more.hidden = true;
  let next = null;
  async function showCases(offset=0) {
    load.disabled = true; more.disabled = true;
    try {
      const params = new URLSearchParams({offset: String(offset)});
      Object.entries(controls).forEach(([name, select]) => { if (select.value) params.set(name, select.value); });
      const result = await api("/api/experiments/cases?" + params);
      if (offset === 0) list.replaceChildren();
      summary.textContent = result.total + " matching saved records · " + result.phase + " · " + result.notice;
      for (const row of result.cases) {
        const item = element("details", "saved-case"), title = element("summary", "", row.condition + " · " + row.outcome.replaceAll("_", " ") + " · " + row.question.question);
        item.append(title, element("p", "", "Expected: " + row.question.expected_answer), element("p", "", "Generated final answer: " + (row.content ?? "No answer generated")));
        item.append(element("p", "routing-note", "Requested: " + row.question.requested_entity + " / " + row.question.requested_relation + " / " + row.question.requested_scope));
        if (row.error) item.append(element("pre", "case-raw", JSON.stringify(row.error, null, 2)));
        item.append(element("h3", "", "Facts supplied to the model"));
        row.supplied_facts.forEach(fact => item.append(element("p", "case-fact", fact.id + " · " + fact.question + " → " + fact.answer + "\nScope: " + fact.scope + "\nSource: " + fact.source)));
        const trace = element("details", "case-trace"); trace.append(element("summary", "", "Raw generated output · explanatory prose is unverified"), element("pre", "case-raw", row.raw_output || "No raw generation recorded")); item.append(trace);
        const generation = row.generation || {};
        item.append(element("p", "routing-note", "Model: " + (generation.model || "No generation") + " · revision " + (generation.revision || "unavailable") + " · pipeline " + (row.pipeline_seconds?.toFixed(2) ?? "unavailable") + " s · prompt SHA256 " + (row.request_sha256 || "unavailable")));
        list.append(item);
      }
      next = result.next_offset; more.hidden = next === null;
    } catch (error) { showError(error); }
    finally { load.disabled = false; more.disabled = false; }
  }
  load.addEventListener("click", () => showCases()); more.addEventListener("click", () => showCases(next));
  panel.append(filters, summary, list, more); target.append(panel);
  await showCases();
}

async function loadResults() {
  const evidence = await api("/api/experiments"), target = $("results-content"); target.replaceChildren();
  const controlled = evidence.kind.startsWith("controlled_");
  $("experiment-kind").textContent = evidence.kind === "controlled_final_benchmark" ? "Controlled final experiment" : controlled ? "Controlled development experiment" : "Development acceptance · real generated answers";
  $("experiment-notice").textContent = evidence.notice;
  target.append(element("p", "result-footnote", "Model: " + evidence.model + " · revision " + evidence.model_revision));
  if (controlled) {
    const report = evidence.report;
    const labels = {R0: "ReasonKE · initial facts", "R-RAG": "ReasonKE · updated retrieval", "R-ALL": "ReasonKE · full approved memory", "G-RAG": "Qwen · updated retrieval", "G-ALL": "Qwen · full approved memory", "R-ORACLE": "ReasonKE · oracle support", "G-ORACLE": "Qwen · oracle support"};
    const cards = element("div", "results-cards");
    cards.append(resultCard(report.independent_lineages, "Independent edit lineages"), resultCard(report.assigned_records, "Assigned records, including errors and abstentions"), resultCard(report.execution_status, "Execution state")); target.append(cards);
    target.append(evidenceTable(["Condition", "Correct / assigned", "Wrong", "Abstain", "Errors", "Unexecuted", "Coverage"], Object.entries(report.conditions).map(([name, condition]) => {
      const metric = condition.overall, counts = metric.counts;
      return [labels[name] || name, counts.correct + "/" + metric.assigned, counts.wrong, counts.abstain, counts.execution_error, counts.unexecuted, metric.coverage === null ? "Undefined" : (100 * metric.coverage).toFixed(1) + "%"];
    })));
    target.append(element("p", "result-footnote", "R0 uses initial facts but is scored against updated targets, so it is a staleness diagnostic. R / G denote the attributed ReasonKE / Qwen checkpoints. RAG supplies the same retrieved facts to both. Oracle uses gold supporting facts on a smaller preselected diagnostic subset."));
    target.append(element("h2", "", "Accuracy by question category"));
    const names = Object.keys(report.conditions).filter(name => !name.endsWith("ORACLE"));
    const categories = Object.keys(report.conditions[names[0]].by_category);
    target.append(evidenceTable(["Category", ...names], categories.map(category => [CATEGORY_LABELS[category] || category, ...names.map(name => {
      const metric = report.conditions[name].by_category[category]; return metric.counts.correct + "/" + metric.assigned;
    })])));
    target.append(element("h2", "", "Warm pipeline latency"));
    target.append(evidenceTable(["Condition", "Generated n", "p50 seconds", "p95 seconds"], names.map(name => {
      const latency = report.conditions[name].overall.generated_latency;
      return [name, latency.n, latency.p50_seconds?.toFixed(2), latency.p95_seconds?.toFixed(2)];
    })));
    target.append(element("p", "result-footnote", "Pipeline timing includes context selection, local worker communication and generation; it excludes loading and the browser network round trip. Cold loading is recorded separately in the raw export."));
    target.append(element("h2", "", "Retrieval and preservation"));
    target.append(evidenceTable(["Condition", "Required facts retrieved", "Complete two-hop support", "Retained previously correct locality answers", "False edited-value endorsements"], names.map(name => {
      const condition = report.conditions[name], retrieval = condition.retrieval, retention = condition.baseline_correct_retention;
      const ratio = (n, d) => d ? n + "/" + d : "Not applicable";
      return [labels[name], retrieval ? ratio(retrieval.fact_recall_numerator, retrieval.fact_recall_denominator) : "Not applicable", retrieval ? ratio(retrieval.two_hop_complete_numerator, retrieval.two_hop_complete_denominator) : "Not applicable", retention ? ratio(retention.correct, retention.denominator) : "Not applicable", ratio(condition.locality.false_correction_applications, condition.locality.assigned)];
    })));
    target.append(element("p", "result-footnote", "Coverage counts answered cases; primary accuracy includes every assigned question. Abstentions receive zero accuracy credit. Preservation only measures answers that R0 already answered correctly; locality accuracy measures factual correctness on all locality questions."));
    const artifacts = new Set(evidence.artifacts || []), figures = element("div", "scientific-figures");
    const titles = {"category-accuracy": "Which questions benefit?", "efficacy-locality": "Correction success and preserved scope", "outcome-composition": "Every assigned outcome", "paired-differences": "Paired uncertainty across edit lineages", latency: "Measured warm request latency"};
    for (const [name, title] of Object.entries(titles)) {
      if (!artifacts.has(name + ".png")) continue;
      const figure = element("figure", "scientific-figure");
      const image = element("img"); image.src = "/api/experiments/artifacts/" + name + ".png"; image.alt = title; image.loading = "lazy";
      const caption = element("figcaption", "", title + " · ");
      const link = element("a", "text-link", "Open exportable SVG"); link.href = "/api/experiments/artifacts/" + name + ".svg"; link.target = "_blank"; link.rel = "noopener";
      caption.append(link); figure.append(image, caption); figures.append(figure);
    }
    target.append(figures);
    const downloads = element("div", "message-actions");
    for (const [name, label] of [["study.pdf", "Open study PDF"], ["REPORT.md", "Read measured report"], ["scorecard.csv", "Export scorecard CSV"], ["figure-manifest.json", "Figure provenance"]]) {
      if (!artifacts.has(name)) continue;
      const link = element("a", "quiet-button", label); link.href = "/api/experiments/artifacts/" + name; link.target = "_blank"; link.rel = "noopener"; downloads.append(link);
    }
    target.append(downloads);
    if (evidence.explanation_audit) {
      const audit = evidence.explanation_audit;
      target.append(element("h2", "", "Explicit subanswer support · separate audit"), element("p", "routing-note", audit.assigned_traces + " assigned " + audit.phase + " traces. This automated parser checks indexed boxed subanswers; surrounding prose stays unassessed. A supported subanswer does not verify a complete explanation or internal reasoning."));
      target.append(evidenceTable(["Condition", "Assigned traces", "Extracted claims", "Supported", "Contradicted", "Unsupported", "Unassessable", "Unassessed prose segments retained"], Object.entries(audit.by_condition).map(([name, value]) => {
        const counts = value.current_fact_counts;
        return [name, value.assigned_traces, value.extracted_indexed_claims, counts.supported, counts.contradicted, counts.unsupported, counts.unassessable, value.unassessable_segments_retained];
      })));
      const link = element("a", "text-link", "Inspect the complete claim-support audit JSON"); link.href = "/api/experiments/artifacts/explanation-audit.json"; link.target = "_blank"; link.rel = "noopener"; target.append(link);
    }
    await addAnswerExplorer(target, report);
  } else {
    const cards = element("div", "results-cards");
    cards.append(resultCard(evidence.profile.matching_final_answers + "/" + evidence.profile.assigned, "Predefined startup final-answer matches"),
      resultCard(evidence.workflow.factual_matches + "/" + evidence.workflow.factual_assigned, "Factual website checks"),
      resultCard(evidence.workflow.assigned, "Website generations · ordinary chat checked separately"),
      resultCard(evidence.workflow.owned_fact_inactive_at_end ? "Confirmed" : "Not confirmed", "Demonstration correction withdrawn")); target.append(cards);
    target.append(element("h2", "", "Complete website workflow"));
    target.append(evidenceTable(["Check", "Expected", "Generated answer", "Evaluation"], evidence.workflow.cases.map(row => [row.id, row.expected ?? "Nonempty only", row.actual, row.factual_quality_evaluated ? (row.passed ? "Exact final match" : "Did not match") : "Ordinary generation · qualitative check"])));
    target.append(element("h2", "", "Startup checks, including two-hop reasoning"));
    target.append(evidenceTable(["Question type", "Expected", "Generated", "Seconds"], evidence.profile.cases.map(row => [row.id, row.expected, row.actual, row.seconds?.toFixed(2)])));
    target.append(element("p", "result-footnote", "Measured cold loading: " + evidence.profile.load_seconds.toFixed(2) + " s. Load peak reserved CUDA memory: " + (evidence.profile.peak_reserved_bytes / 1024 ** 3).toFixed(2) + " GiB. These are session measurements."));
    const limitations = element("section", "tradeoff-note caution"); limitations.append(element("h3", "", "What these results establish"), element("p", "", evidence.notice));
    evidence.limitations.forEach(text => limitations.append(element("p", "", text))); target.append(limitations);
  }
  state.resultsLoaded = true;
}

$("current-results").addEventListener("click", () => loadResults().catch(showError));
$("archived-results").addEventListener("click", () => {
  $("experiment-kind").textContent = "Archived visual-model study";
  $("experiment-notice").textContent = "BLIP-2 experiment with a different model, dataset and protocol. Its scores are not current text-chat accuracy.";
  loadArchivedResults().catch(showError);
});
$("download-results").addEventListener("click", async () => {
  try {
    const result = await api("/api/experiments"), url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"}));
    const link = element("a"); link.href = url; link.download = "editguard-current-experiment.json"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { showError(error); }
});

async function loadArchivedResults() {
  const received = await api("/api/results");
  const report = received.report || received;
  const target = $("results-content"); target.replaceChildren();
  if (report.execution_status !== "completed" || !report.aggregate?.scores) { target.append(element("p", "muted", "The saved visual benchmark is unavailable. No numerical results are being invented or substituted.")); return; }
  const scores = report.aggregate.scores;
  const cells = Object.values(scores).flatMap(item => Object.values(item));
  const sum = key => cells.reduce((total, row) => total + (typeof row[key] === "number" ? row[key] : 0), 0);
  const cards = element("div", "results-cards");
  cards.append(resultCard(String(report.completed_edit_units ?? report.worker_report?.completed_edit_units ?? "—"), "Independent edit episodes"), resultCard(String(sum("denominator")), "Assigned answer records"), resultCard(String(sum("errors_or_unexecuted")), "Execution errors / unexecuted"), resultCard(String(sum("abstentions")), "Empty answers · counted as incorrect"));
  target.append(cards);
  const panel = element("section", "result-panel");
  panel.append(element("h2", "", "Target success and unrelated-answer accuracy"), element("p", "", "Counts use strict whole-answer matching. Every category has its assigned denominator, including empty and failed answers. The unchanged base is a diagnostic reference for counterfactual targets."));
  const tableWrap = element("div", "result-table-wrap"); const table = element("table");
  const header = element("tr"); ["METHOD", "DIRECT", "PARAPHRASE", "OTHER IMAGE", "TEXT LOCALITY", "VISUAL LOCALITY"].forEach(label => header.append(element("th", "", label))); const thead = element("thead"); thead.append(header); table.append(thead);
  const names = {base:"Unchanged base", prompt_only:"Prompt-only correction", updated_rag:"Updated visual RAG", mend_context:"MEND · adapted", ft_last_layer:"Fine-tuning · adapted"};
  const categories = ["direct", "paraphrase", "alternate_image", "text_locality", "visual_locality"];
  const tbody = element("tbody");
  for (const [method, row] of Object.entries(scores)) {
    const tr = element("tr", method === "updated_rag" ? "highlight-row" : ""); tr.append(element("td", "", names[method] || method));
    categories.forEach(category => { const metric = row[category]; tr.append(element("td", "", metric ? metric.correct + "/" + metric.denominator : "Unavailable")); }); tbody.append(tr);
  }
  table.append(tbody); tableWrap.append(table); panel.append(tableWrap); target.append(panel);
  const notes = element("div", "tradeoff-grid");
  const locality = element("section", "tradeoff-note caution"); locality.append(element("h3", "", "A successful edit can damage other answers."), element("p", "", "Fine-tuning matched 60/60 edited targets but 0/40 unrelated targets, losing all five previously correct locality answers. The other 35 locality answers were already incorrect before editing. MEND retained those five, a very small preservation denominator."));
  const rag = element("section", "tradeoff-note"); rag.append(element("h3", "", "RAG is a useful default, not a universal winner."), element("p", "", "Updated RAG and prompt-only each matched 54/60 edited targets. RAG falsely applied a correction to 13/40 unrelated cases. These visual results do not calibrate or establish accuracy for this app’s multi-correction conversational retriever.")); notes.append(locality, rag); target.append(notes);
  target.append(element("p", "result-footnote", "Limits: 20 nonrandom edit units, controlled counterfactual targets and automated label audit without independent human review. Published paper metrics are separately attributed. No claim of general superiority, arbitrary-edit safety or complete reproduction of the original paper."));
  const link = element("a", "text-link", "Open the preserved visual parameter-editing workbench notebook ↗"); link.href = "https://colab.research.google.com/github/ShivamRaghav7/verity/blob/d4aa5735de5340921bf00103b53326542397f51a/notebooks/03_visual_editing_workbench.ipynb"; link.target = "_blank"; link.rel = "noopener"; target.append(link);
  state.resultsLoaded = true;
}

document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => { $("chat-input").value = button.dataset.prompt; resizeComposer(); $("chat-input").focus(); }));
$("starter-knowledge").addEventListener("click", () => showView("knowledge"));
$("chat-edit-knowledge").addEventListener("click", () => showView("knowledge"));
$("new-chat").addEventListener("click", () => { clearError(); createChat().catch(showError); });
$("mobile-new-chat").addEventListener("click", () => { clearError(); createChat().catch(showError); });
$("mobile-chat-select").addEventListener("change", event => { if (event.target.value && !state.busy) openChat(event.target.value).catch(showError); });
$("chat-form").addEventListener("submit", event => sendMessage(event).catch(showError));
$("chat-input").addEventListener("input", resizeComposer);
$("chat-input").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); } });
$("image-input").addEventListener("change", event => { clearError(); attachImage(event.target.files[0]).catch(showError); });
$("remove-attachment").addEventListener("click", clearAttachment);
$("correction-form").addEventListener("submit", event => publishCorrection(event).catch(showError));
$("show-withdrawn").addEventListener("change", renderCorrections);
$("correction-search").addEventListener("input", renderCorrections);
$("cancel-replacement").addEventListener("click", resetReplacement);
$("close-compare").addEventListener("click", () => $("compare-dialog").close());
$("compare-baseline-mode").addEventListener("change", event => {
  const historical = event.target.value === "revision";
  $("compare-baseline-revision").disabled = !historical;
  $("compare-baseline-revision").max = state.revision;
  if (historical) $("compare-baseline-revision").value = Math.max(0, state.revision - 1);
});
$("rerun-comparison").addEventListener("click", () => { if (state.compareQuestion) compare(state.compareQuestion).catch(showError); });
$("compare-dialog").addEventListener("click", event => { if (event.target === $("compare-dialog")) { const bounds = $("compare-dialog").getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $("compare-dialog").close(); } });
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
document.addEventListener("keydown", event => { if (event.key.toLowerCase() === "n" && !event.ctrlKey && !event.metaKey && !event.altKey && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) && !$("compare-dialog").open && !state.busy) { event.preventDefault(); createChat().catch(showError); } });

async function initialize() {
  const initialView = location.hash.slice(1) || "chat";
  await refreshStatus();
  await Promise.all([loadChats(), loadCorrections()]);
  const previous = localStorage.getItem("editguard-chat-id");
  if (previous && state.chats.some(chat => String(chat.id) === previous)) await openChat(previous);
  showView(initialView);
}
initialize().catch(showError);
setInterval(refreshStatus, 5000);
