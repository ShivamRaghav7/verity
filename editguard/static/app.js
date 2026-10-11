"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const state = { facts: [], edits: [], events: [], conflicts: [], revision: null, health: null, initialized: false, busy: false, lastAnswerRevision: null };
  const viewNames = { assistant: "SUPPORT ASSISTANT", corrections: "CORRECTION EDITOR", history: "VERSION HISTORY", conflicts: "RESOLVE CONFLICTS", evaluation: "EVALUATION LAB" };
  const relationNames = { device_limit: "Device limit", price_monthly: "Monthly price", storage_gb: "Storage allowance", support_hours: "Support hours" };

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }
  function printable(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
  function readable(value) { return String(value || "").replaceAll("_", " "); }
  function factName(fact) {
    if (!fact) return "Unknown fact";
    return `${fact.entity_name || fact.entity_id} · ${fact.plan || "All plans"} · ${fact.version || "All versions"} · ${relationNames[fact.relation] || readable(fact.relation)}`;
  }
  function findFact(id) { return state.facts.find(fact => fact.fact_id === id); }
  function dateWindow(record) { return `${record.effective_from || "Unspecified"} → ${record.effective_to || "open-ended"} (end exclusive)`; }
  function formatTimestamp(value) {
    if (!value) return "Time unavailable";
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? String(value) : new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Kolkata" }).format(date) + " IST";
  }
  function notify(message, kind = "success") {
    const box = byId("global-message");
    box.textContent = message;
    box.className = "message" + (kind === "success" ? "" : ` ${kind}`);
    box.hidden = false;
    box.setAttribute("role", kind === "error" ? "alert" : "status");
  }
  function clearNotice() { byId("global-message").hidden = true; }
  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
    let result;
    try { result = await response.json(); } catch { throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`); }
    if (!response.ok) {
      const error = new Error(typeof result.detail === "string" ? result.detail : printable(result.detail || `Request failed (HTTP ${response.status}).`));
      error.status = response.status;
      throw error;
    }
    return result;
  }
  function selectView(view, updateHash = true) {
    if (!viewNames[view]) view = "assistant";
    for (const [name] of Object.entries(viewNames)) byId(`view-${name}`).hidden = name !== view;
    document.querySelectorAll(".nav-item").forEach(button => {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    byId("view-breadcrumb").textContent = viewNames[view];
    if (updateHash && location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  }
  function makeOption(value, label) {
    const option = node("option", label);
    option.value = value;
    return option;
  }
  function fillSelect(id, records, emptyLabel) {
    const select = byId(id);
    const current = select.value;
    select.replaceChildren(makeOption("", emptyLabel));
    records.forEach(record => select.append(makeOption(record.value, record.label)));
    if (records.some(record => String(record.value) === current)) select.value = current;
  }
  function populateSelectors() {
    const entities = new Map();
    const plans = new Set();
    const versions = new Set();
    state.facts.forEach(fact => {
      entities.set(fact.entity_id, fact.entity_name || fact.entity_id);
      if (fact.plan) plans.add(fact.plan);
      if (fact.version) versions.add(fact.version);
    });
    fillSelect("ask-entity", [...entities].map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label)), "Infer from question");
    fillSelect("ask-plan", [...plans].sort().map(value => ({ value, label: value })), "Infer from question");
    fillSelect("ask-version", [...versions].sort().map(value => ({ value, label: value })), "Infer from question");
    fillSelect("correction-fact", state.facts.map(fact => ({ value: fact.fact_id, label: factName(fact) })), "Select a catalogue fact");
    updatePredecessors();
    if (!state.initialized) {
      const demo = state.facts.find(fact => fact.entity_name === "Nimbus" && fact.plan === "Basic" && fact.version === "v2" && fact.relation === "device_limit");
      if (demo) {
        byId("ask-entity").value = demo.entity_id;
        byId("ask-plan").value = demo.plan;
        byId("ask-version").value = demo.version;
        byId("correction-fact").value = demo.fact_id;
        selectedFactChanged();
      }
      state.initialized = true;
    }
  }
  function detail(container, label, value) {
    const item = node("div", undefined, "fact-detail");
    item.append(node("label", label), node("span", value || "—"));
    container.append(item);
  }
  function factDetails(fact, includeSource = true) {
    const details = node("div", undefined, "fact-details");
    detail(details, "Product", fact.entity_name || fact.entity_id);
    detail(details, "Plan", fact.plan || "All plans");
    detail(details, "Version", fact.version || "All versions");
    detail(details, "Property", relationNames[fact.relation] || readable(fact.relation));
    detail(details, "Initial value", [fact.value, fact.unit].filter(Boolean).join(" "));
    detail(details, "Effective window", dateWindow(fact));
    if (includeSource) detail(details, "Catalogue source", fact.source);
    return details;
  }
  function updatePredecessors() {
    const id = byId("correction-fact").value;
    fillSelect("supersedes", state.edits.filter(edit => edit.fact_id === id && edit.status === "approved").map(edit => ({ value: edit.edit_id, label: `${edit.corrected_value} · from ${edit.effective_from} · ${edit.edit_id}` })), "No predecessor correction");
  }
  function selectedFactChanged() {
    const fact = findFact(byId("correction-fact").value);
    const preview = byId("fact-preview");
    preview.replaceChildren();
    preview.hidden = !fact;
    if (fact) {
      preview.append(factDetails(fact));
      byId("original-value").value = fact.value;
    } else byId("original-value").value = "";
    byId("supersedes").value = "";
    updatePredecessors();
  }
  function setHealth() {
    const health = state.health;
    const live = health && health.status === "ok";
    const dot = byId("connection-dot");
    dot.className = `status-dot ${health ? "online" : "offline"}`;
    byId("connection-state").textContent = health ? (live ? "Local API connected" : `Local API · ${readable(health.status)}`) : "Local API unavailable";
    byId("revision-label").textContent = state.revision === null ? "Revision —" : `Revision ${state.revision}`;
    if (state.lastAnswerRevision !== null && state.lastAnswerRevision !== state.revision) {
      byId("answer-revision").textContent = `COMPARISON REVISION ${state.lastAnswerRevision} · ASK AGAIN FOR REVISION ${state.revision}`;
    }
    const models = health?.models || {};
    byId("scope-model-state").textContent = models.scope ? readable(models.scope) : "Not trained";
    byId("answer-model-state").textContent = models.answer ? readable(models.answer) : "Not trained";
  }
  async function loadData(reportErrors = true) {
    const endpoints = ["/api/health", "/api/facts", "/api/edits", "/api/conflicts"];
    const results = await Promise.allSettled(endpoints.map(path => api(path)));
    const failures = [];
    results.forEach((result, index) => {
      if (result.status === "rejected") { failures.push(`${endpoints[index]}: ${result.reason.message}`); if (index === 0) state.health = null; return; }
      const data = result.value;
      if (index === 0) state.health = data;
      if (index === 1) state.facts = data.facts || [];
      if (index === 2) { state.edits = data.edits || []; state.events = data.events || []; }
      if (index === 3) state.conflicts = data.conflicts || [];
      if (typeof data.revision === "number") state.revision = Math.max(state.revision ?? 0, data.revision);
    });
    setHealth();
    populateSelectors();
    renderHistory();
    renderConflicts();
    if (failures.length && reportErrors) notify(`Unable to load all workspace data. ${failures.join(" ")}`, "error");
    return failures.length === 0;
  }
  function emptyState(title, message, symbol = "○") {
    const box = node("div", undefined, "empty-state");
    box.append(node("span", symbol, "empty-state-symbol"), node("strong", title), node("p", message));
    return box;
  }
  function actionButton(label, action, editId, dangerous = false) {
    const button = node("button", label, `button button-small ${dangerous ? "button-danger" : "button-secondary"}`);
    button.type = "button";
    button.dataset.action = action;
    button.dataset.editId = editId;
    button.addEventListener("click", () => mutateEdit(editId, action, button));
    return button;
  }
  function editActions(edit) {
    const actions = node("div", undefined, "record-actions");
    if (edit.status === "draft") actions.append(actionButton("Approve correction", "approve", edit.edit_id), actionButton("Withdraw draft", "withdraw", edit.edit_id, true));
    if (edit.status === "approved") actions.append(actionButton("Roll back / withdraw", "withdraw", edit.edit_id, true));
    return actions;
  }
  function correctionCard(edit, compact = false) {
    const fact = findFact(edit.fact_id) || edit;
    const card = node("article", undefined, compact ? "conflict-correction" : "panel record-card");
    const top = node("div", undefined, "record-top");
    top.append(node("h3", factName(fact)), node("span", readable(edit.status), `pill pill-${edit.status}`));
    const values = node("div", undefined, "record-value");
    values.append(node("span", edit.original_value, "old"), node("span", "→", "value-arrow"), node("span", edit.corrected_value, "new"), node("span", fact.unit || "", "muted"));
    card.append(top, values, node("p", `Effective: ${dateWindow(edit)}`, "record-context"), node("p", `Source: ${edit.source || "Source unavailable"}`, "record-context"));
    if (edit.supersedes) card.append(node("p", `Predecessor: ${edit.supersedes}`, "record-context"));
    if (edit.reason) card.append(node("p", edit.reason, "record-reason"));
    const bottom = node("div", undefined, "record-bottom");
    bottom.append(node("span", edit.edit_id, "record-id"), editActions(edit));
    card.append(bottom);
    return card;
  }
  function renderHistory() {
    byId("history-count").textContent = state.edits.length;
    byId("history-total").textContent = state.edits.length;
    for (const [id, status] of [["history-approved", "approved"], ["history-drafts", "draft"], ["history-withdrawn", "withdrawn"]]) byId(id).textContent = state.edits.filter(edit => edit.status === status).length;
    const list = byId("history-list");
    list.replaceChildren();
    if (!state.edits.length) list.append(emptyState("A clean starting point.", "Create a correction draft to start the version history.", "↶"));
    [...state.edits].reverse().forEach(edit => list.append(correctionCard(edit)));
    const audit = byId("audit-list");
    audit.replaceChildren();
    if (!state.events.length) audit.append(node("p", "No lifecycle events recorded yet.", "muted"));
    [...state.events].reverse().forEach(event => {
      const row = node("div", undefined, "audit-event");
      const time = node("time", formatTimestamp(event.created_at || event.timestamp || event.at));
      if (event.created_at || event.timestamp || event.at) time.dateTime = event.created_at || event.timestamp || event.at;
      const description = [readable(event.action || event.event || event.event_type || "Lifecycle event"), event.edit_id || event.details?.edit_id, event.actor ? `by ${event.actor}` : "", event.revision !== undefined ? `revision ${event.revision}` : ""].filter(Boolean).join(" · ");
      row.append(time, node("p", description));
      audit.append(row);
    });
  }
  function renderConflicts() {
    byId("conflict-count").textContent = state.conflicts.length;
    const list = byId("conflicts-list");
    list.replaceChildren();
    if (!state.conflicts.length) list.append(emptyState("No unresolved contradictions.", "The current correction memory has no reported overlapping conflicts.", "◇"));
    state.conflicts.forEach(conflict => {
      const card = node("article", undefined, "panel conflict-card");
      const top = node("div", undefined, "record-top");
      top.append(node("h3", factName(findFact(conflict.fact_id))), node("span", "Resolution required", "pill pill-conflict"));
      card.append(top, node("p", conflict.reason || "Overlapping corrections endorse incompatible values.", "record-reason"));
      const records = node("div", undefined, "conflict-corrections");
      (conflict.edit_ids || []).forEach(id => {
        const edit = state.edits.find(candidate => candidate.edit_id === id);
        if (edit) records.append(correctionCard(edit, true)); else records.append(node("p", `Correction ${id} · details unavailable`, "record-context"));
      });
      card.append(records, node("p", "Review source and effective dates. Withdrawing a conflicting draft retains the audit record; approval remains a separate action.", "field-hint"));
      list.append(card);
    });
  }
  function requireToken() {
    const token = byId("admin-token").value.trim();
    if (!token) {
      notify("Enter the configured administrator token in the correction editor before changing correction memory.", "warning");
      selectView("corrections");
      byId("admin-token").focus();
      return null;
    }
    return token;
  }
  function setBusy(busy) {
    state.busy = busy;
    document.querySelectorAll("[data-action], #create-button").forEach(button => { button.disabled = busy; });
  }
  function renderDraftReview(edit) {
    const panel = byId("draft-review");
    panel.hidden = false;
    panel.replaceChildren(node("h3", edit.status === "draft" ? "Draft ready for review" : "Correction lifecycle updated"), node("span", readable(edit.status), `pill pill-${edit.status}`));
    panel.append(node("p", `${edit.original_value} → ${edit.corrected_value}`, "record-value"), node("p", factName(findFact(edit.fact_id) || edit), "record-context"), node("p", `Effective: ${dateWindow(edit)}`, "record-context"), node("p", `Source: ${edit.source}`, "record-context"), node("p", edit.edit_id, "record-id"));
    if (edit.status === "draft") panel.append(node("p", "Review these details, then approve to activate. Conflicting scopes must be resolved first.", "field-hint"));
    panel.append(editActions(edit));
  }
  async function mutateEdit(editId, action, button) {
    if (state.busy) return;
    const token = requireToken();
    if (!token) return;
    clearNotice();
    setBusy(true);
    const previousLabel = button.textContent;
    button.textContent = action === "approve" ? "Approving…" : "Withdrawing…";
    try {
      const edit = await api(`/api/edits/${encodeURIComponent(editId)}/${action}`, { method: "POST", headers: { "X-Admin-Token": token } });
      await loadData();
      renderDraftReview(edit);
      notify(action === "approve" ? "Correction approved. Compare answers to inspect the current effective behavior." : "Correction withdrawn. Applicable predecessor facts are restored for subsequent questions.");
    } catch (error) {
      await loadData(false);
      notify(error.status === 409 ? `Conflict prevents this action. ${error.message}` : error.message, "error");
      if (error.status === 409) selectView("conflicts");
    } finally { button.textContent = previousLabel; setBusy(false); }
  }
  async function createCorrection(event) {
    event.preventDefault();
    if (state.busy) return;
    const token = requireToken();
    if (!token) return;
    const payload = { fact_id: byId("correction-fact").value, original_value: byId("original-value").value.trim(), corrected_value: byId("corrected-value").value.trim(), source: byId("correction-source").value.trim(), reason: byId("correction-reason").value.trim(), effective_from: byId("effective-from").value };
    if (byId("effective-to").value) payload.effective_to = byId("effective-to").value;
    if (byId("supersedes").value) payload.supersedes = byId("supersedes").value;
    clearNotice();
    setBusy(true);
    byId("create-button").textContent = "Creating draft…";
    try {
      const edit = await api("/api/edits", { method: "POST", headers: { "X-Admin-Token": token }, body: JSON.stringify(payload) });
      await loadData();
      renderDraftReview(edit);
      notify("Correction saved as a draft. Review and approve it explicitly before activation.");
    } catch (error) { notify(error.message, "error"); }
    finally { byId("create-button").textContent = "Create correction draft ＋"; setBusy(false); }
  }
  function renderAnswer(result) {
    byId("answer-placeholder").hidden = true;
    byId("answer-results").hidden = false;
    byId("before-answer").textContent = result.before_answer || "The base assistant could not resolve a unique answer.";
    byId("after-answer").textContent = result.answer || "Additional information is required.";
    const status = byId("answer-status");
    status.textContent = result.status === "answered" ? "Answered · structured" : result.status === "conflict" ? "Conflict · resolve required" : "Clarification needed";
    status.className = `pill pill-${result.status || "clarification"}`;
    byId("answer-latency").textContent = typeof result.latency_ms === "number" ? `${result.latency_ms.toFixed(1)} ms · API processing` : "";
    byId("answer-revision").textContent = `REVISION ${printable(result.revision)}`;
    byId("answer-explanation").textContent = [result.explanation, result.route ? `Route: ${printable(result.route)}` : ""].filter(Boolean).join(" ");
    const fact = byId("answer-fact");
    fact.replaceChildren();
    if (result.fact) fact.append(factDetails(result.fact));
    const edits = byId("answer-edits");
    edits.replaceChildren();
    (result.applicable_corrections || []).forEach(edit => {
      const item = node("div", undefined, "evidence-item");
      if (typeof edit === "string") item.append(node("strong", edit));
      else item.append(node("strong", `Correction ${edit.edit_id} · ${edit.corrected_value}`), node("p", `Source: ${edit.source || "Unavailable"} · ${dateWindow(edit)}`));
      edits.append(item);
    });
    if (!(result.applicable_corrections || []).length) edits.append(node("p", result.status === "answered" ? "No approved correction applies. The current answer uses the initial catalogue fact." : "No corrected answer is endorsed until the ambiguity or conflict is resolved.", "muted"));
    state.lastAnswerRevision = result.revision;
  }
  async function askQuestion(event) {
    event.preventDefault();
    const button = byId("ask-button");
    if (button.disabled) return;
    const payload = { question: byId("question").value.trim(), as_of: byId("ask-date").value, mode: "structured" };
    if (!payload.question) { byId("question").focus(); return; }
    for (const [field, id] of [["entity_id", "ask-entity"], ["plan", "ask-plan"], ["version", "ask-version"]]) if (byId(id).value) payload[field] = byId(id).value;
    clearNotice();
    button.disabled = true;
    button.textContent = "Resolving scoped facts…";
    const started = performance.now();
    try {
      const result = await api("/api/ask", { method: "POST", body: JSON.stringify(payload) });
      renderAnswer(result);
      const elapsed = performance.now() - started;
      byId("answer-latency").textContent += `${byId("answer-latency").textContent ? " · " : ""}${elapsed.toFixed(0)} ms browser round trip`;
      if (typeof result.revision === "number") { state.revision = result.revision; setHealth(); }
    } catch (error) { notify(error.message, "error"); }
    finally { button.disabled = false; button.textContent = "Compare answers →"; }
  }
  async function runEvaluation() {
    const button = byId("run-evaluation");
    button.disabled = true;
    button.textContent = "Running regression…";
    clearNotice();
    try {
      const evaluation = await api("/api/evaluation");
      const status = byId("evaluation-status");
      status.textContent = readable(evaluation.status || "Complete");
      status.className = `pill ${evaluation.status === "failed" ? "pill-failed" : "pill-teal"}`;
      byId("evaluation-disclaimer").textContent = evaluation.disclaimer || "Actual deterministic engineering regression results. These are not trained-model evaluation results.";
      const metrics = byId("evaluation-metrics");
      metrics.replaceChildren();
      (evaluation.metrics || []).forEach(metric => {
        const card = node("article", undefined, "metric-card");
        const value = typeof metric.value === "number" ? (Number.isInteger(metric.value) ? metric.value : Number(metric.value.toFixed(3))) : printable(metric.value);
        card.append(node("label", readable(metric.name)), node("strong", `${value}${metric.unit ? ` ${metric.unit}` : ""}`));
        if (metric.denominator !== undefined) card.append(node("small", `Denominator: ${printable(metric.denominator)}`));
        if (metric.description) card.append(node("small", metric.description));
        metrics.append(card);
      });
      const cases = byId("evaluation-cases");
      cases.replaceChildren();
      if (!(evaluation.cases || []).length) cases.append(node("p", "No case-level regression results returned.", "muted"));
      else {
        const wrapper = node("div", undefined, "evaluation-table-wrapper");
        const table = node("table", undefined, "evaluation-table");
        const head = node("thead");
        const headings = node("tr");
        ["Regression case", "Outcome", "Expected", "Actual"].forEach(label => { const cell = node("th", label); cell.scope = "col"; headings.append(cell); });
        head.append(headings);
        const body = node("tbody");
        evaluation.cases.forEach(test => {
          const row = node("tr");
          const outcome = node("td");
          outcome.append(node("span", test.passed ? "Passed" : "Failed", `pill ${test.passed ? "pill-teal" : "pill-failed"}`));
          row.append(node("td", test.name || test.case_id), outcome, node("td", printable(test.expected)), node("td", printable(test.actual)));
          body.append(row);
        });
        table.append(head, body); wrapper.append(table); cases.append(wrapper);
      }
    } catch (error) { notify(error.message, "error"); byId("evaluation-status").textContent = "Request failed"; }
    finally { button.disabled = false; button.textContent = "Run slice regression →"; }
  }
  async function exportData() {
    const button = byId("export-button");
    button.disabled = true;
    clearNotice();
    try {
      const exported = await api("/api/export");
      const url = URL.createObjectURL(new Blob([JSON.stringify(exported, null, 2)], { type: "application/json" }));
      const anchor = node("a");
      anchor.href = url;
      anchor.download = `editguard-export-revision-${state.revision ?? "unknown"}.json`;
      document.body.append(anchor); anchor.click(); anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify("Workspace data and available slice evaluation exported as JSON.");
    } catch (error) { notify(error.message, "error"); }
    finally { button.disabled = false; }
  }

  document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => { clearNotice(); selectView(button.dataset.view); }));
  window.addEventListener("hashchange", () => selectView(location.hash.slice(1), false));
  document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => {
    byId("question").value = button.dataset.question;
    if (button.dataset.question.includes("Pro")) byId("ask-plan").value = "Pro";
    else if (button.dataset.question.includes("Basic")) byId("ask-plan").value = "Basic";
    byId("question").focus();
  }));
  document.querySelectorAll(".refresh-data").forEach(button => button.addEventListener("click", async () => {
    clearNotice(); button.disabled = true;
    try { if (await loadData()) notify("Workspace records refreshed."); } finally { button.disabled = false; }
  }));
  byId("ask-form").addEventListener("submit", askQuestion);
  byId("correction-form").addEventListener("submit", createCorrection);
  byId("correction-fact").addEventListener("change", selectedFactChanged);
  byId("supersedes").addEventListener("change", () => {
    const edit = state.edits.find(candidate => candidate.edit_id === byId("supersedes").value);
    const fact = findFact(byId("correction-fact").value);
    byId("original-value").value = edit ? edit.corrected_value : fact?.value || "";
  });
  byId("run-evaluation").addEventListener("click", runEvaluation);
  byId("export-button").addEventListener("click", exportData);
  selectView(location.hash.slice(1) || "assistant", false);
  loadData().catch(error => notify(`Workspace initialization failed: ${error.message}`, "error"));
})();
