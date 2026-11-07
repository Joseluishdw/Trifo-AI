// NeuroZero 3.0 (Trifo-AI) — Cyber-Lab Frontend (Zero Emojis · SVG Icons · High-DPI Canvas · Split View)

const appSidebar = document.getElementById("appSidebar");
const btnToggleSidebar = document.getElementById("btnToggleSidebar");
const mainWorkspace = document.getElementById("mainWorkspace");
const activeViewTitle = document.getElementById("activeViewTitle");
const btnSplitView = document.getElementById("btnSplitView");
const splitResizer = document.getElementById("splitResizer");

const systemStatusPill = document.getElementById("systemStatusPill");
const systemStatusText = document.getElementById("systemStatusText");

const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const btnAttachDoc = document.getElementById("btnAttachDoc");
const btnUploadFromKnowledge = document.getElementById("btnUploadFromKnowledge");
const fileUploadInput = document.getElementById("fileUploadInput");
const teachingBanner = document.getElementById("teachingBanner");
const pendingStimulusText = document.getElementById("pendingStimulusText");

const autoWebToggle = document.getElementById("autoWebToggle");
const autoSelfPlayToggle = document.getElementById("autoSelfPlayToggle");
const curiosityToggle = document.getElementById("curiosityToggle");
const btnTrainEpochs = document.getElementById("btnTrainEpochs");
const btnManualSelfPlay = document.getElementById("btnManualSelfPlay");

const btnExportBackup = document.getElementById("btnExportBackup");
const btnImportBackup = document.getElementById("btnImportBackup");
const backupUploadInput = document.getElementById("backupUploadInput");

const btnResetMenu = document.getElementById("btnResetMenu");
const resetMenu = document.getElementById("resetMenu");
const btnResetSeed = document.getElementById("btnResetSeed");
const btnResetBlank = document.getElementById("btnResetBlank");

const metricParams = document.getElementById("metricParams");
const metricVocab = document.getElementById("metricVocab");
const metricRagTasks = document.getElementById("metricRagTasks");
const metricSteps = document.getElementById("metricSteps");
const metricLoss = document.getElementById("metricLoss");
const sideMetricParams = document.getElementById("sideMetricParams");
const sideMetricLoss = document.getElementById("sideMetricLoss");

const epochsLabel = document.getElementById("epochsLabel");
const archLabel = document.getElementById("archLabel");
const attnTokens = document.getElementById("attnTokens");
const headSwitcher = document.getElementById("headSwitcher");
const workingMemBox = document.getElementById("workingMemBox");

const ragDocsList = document.getElementById("ragDocsList");
const wikiResearchForm = document.getElementById("wikiResearchForm");
const wikiQueryInput = document.getElementById("wikiQueryInput");
const btnWikiGo = document.getElementById("btnWikiGo");

const quickTaskForm = document.getElementById("quickTaskForm");
const quickTaskInput = document.getElementById("quickTaskInput");
const quickNoteForm = document.getElementById("quickNoteForm");
const quickNoteInput = document.getElementById("quickNoteInput");

const kanbanPendingList = document.getElementById("kanbanPendingList");
const kanbanDoneList = document.getElementById("kanbanDoneList");
const kanbanPendingCount = document.getElementById("kanbanPendingCount");
const kanbanDoneCount = document.getElementById("kanbanDoneCount");
const notesBoardList = document.getElementById("notesBoardList");
const tasksCountBadge = document.getElementById("tasksCountBadge");
const notesCountBadge = document.getElementById("notesCountBadge");

const factsFilterInput = document.getElementById("factsFilterInput");
const factsCountBadge = document.getElementById("factsCountBadge");
const factsList = document.getElementById("factsList");

const selfTeachList = document.getElementById("selfTeachList");
const episodesList = document.getElementById("episodesList");
const vocabCloud = document.getElementById("vocabCloud");
const neighborsResult = document.getElementById("neighborsResult");
const directTeachForm = document.getElementById("directTeachForm");

const correctionModal = document.getElementById("correctionModal");
const correctionForm = document.getElementById("correctionForm");
const modalEpisodeId = document.getElementById("modalEpisodeId");
const modalCorrectedText = document.getElementById("modalCorrectedText");
const btnCancelModal = document.getElementById("btnCancelModal");

const nnViewport = document.getElementById("nnViewport");
const nnCanvas = document.getElementById("nnCanvas");
const nnCtx = nnCanvas.getContext("2d");

const lossViewport = document.getElementById("lossViewport");
const lossCanvas = document.getElementById("lossCanvas");
const lossCtx = lossCanvas.getContext("2d");

const canvasTooltip = document.getElementById("canvasTooltip");

let currentView = "chat";
let isSplitMode = false;
let selectedAttentionHead = "combined";
let latestHeadAttention = {};
let latestFallbackAttention = [];
let latestActivations = null;
let latestLossHistory = [];
let cachedFacts = [];
let cachedRules = [];

let nnSize = { w: 400, h: 240 };
let lossSize = { w: 400, h: 240 };
let nnHitNodes = [];
let hoveredNnNode = null;
let lossHitPoints = [];
let hoveredLossIdx = null;

// BIBLIOTECA DE ÍCONOS VECTORIALES MINIMALISTAS (STROKE 1.75 · CERO EMOJIS)
const SVG_ICONS = {
  thumbUp: `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`,
  thumbDown: `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>`,
  edit: `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>`,
  check: `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
  close: `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`,
  globe: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>`,
  file: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`,
  table: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line><line x1="9" y1="3" x2="9" y2="21"></line></svg>`,
  rule: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 3 21 3 21 8"></polyline><line x1="4" y1="20" x2="21" y2="3"></line><polyline points="21 16 21 21 16 21"></polyline><line x1="15" y1="15" x2="21" y2="21"></line><line x1="4" y1="4" x2="9" y2="9"></line></svg>`,
  node: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><line x1="3" y1="12" x2="9" y2="12"></line><line x1="15" y1="12" x2="21" y2="12"></line></svg>`,
  note: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>`,
};

const VIEW_TITLES = {
  chat: "Chat & Agente",
  inspector: "Inspector Neuronal en Vivo",
  knowledge: "Centro de Conocimiento y Productividad",
  split: "Split View · Chat + Inspector en Vivo",
};

const MODE_LABELS = {
  neural_hybrid: "MHSA · TRANSFORMER",
  auto_self_taught: "SELF-TAUGHT",
  autonomous_synthesis: "RERANKER K=5",
  knowledge_graph: "SEMANTIC GRAPH",
  logical_reasoning: "SYMBOLIC LOGIC",
  logical_contradiction: "CONFLICT DETECTED",
  document_rag: "RAG VECTOR CACHE",
  web_research: "WIKIPEDIA SYNC",
  agent_tool: "MULTI-TOOL AGENT",
  curiosity_ask: "MANUAL PROMPT",
  learned_from_curiosity: "SUPERVISED LEARN",
  explicit_teaching: "RULE BOUND",
  chat_correction: "WEIGHTS UPDATED",
  backup_restore: "STATE SNAPSHOT",
};

const HEAD_LABELS = {
  combined: "Combinada (Pooling)",
  head_0: "Cabeza H1",
  head_1: "Cabeza H2",
  head_2: "Cabeza H3",
  head_3: "Cabeza H4",
};

function setSystemStatus(busy, text = "IDLE · READY") {
  systemStatusPill.dataset.state = busy ? "busy" : "idle";
  systemStatusText.textContent = text;
}

// Elimina pictogramas emoji residuales provenientes del backend conservando símbolos matemáticos y flechas
function stripPictographicEmojis(str) {
  return String(str).replace(/\p{Extended_Pictographic}\uFE0F?[ ]?/gu, "");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatRichText(raw) {
  const clean = stripPictographicEmojis(raw).replace(/\n{3,}/g, "\n\n").trim();
  let html = escapeHtml(clean);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  return html;
}

// =========================================================================
// NAVEGACIÓN DE 3 ZONAS, SIDEBAR COLAPSABLE Y SPLIT VIEW REDIMENSIONABLE
// =========================================================================
function setWorkspaceMode(targetView, splitOverride = null) {
  if (splitOverride !== null) {
    isSplitMode = splitOverride;
  } else {
    isSplitMode = false;
    currentView = targetView;
  }

  document.querySelectorAll(".sidebar-nav .nav-item[data-view]").forEach((btn) => {
    btn.classList.toggle("active", !isSplitMode && btn.dataset.view === currentView);
  });
  btnSplitView.classList.toggle("active", isSplitMode);

  mainWorkspace.classList.remove("mode-chat", "mode-inspector", "mode-knowledge", "mode-split");
  if (isSplitMode) {
    mainWorkspace.classList.add("mode-split");
    activeViewTitle.textContent = VIEW_TITLES.split;
  } else {
    mainWorkspace.classList.add(`mode-${currentView}`);
    activeViewTitle.textContent = VIEW_TITLES[currentView] || VIEW_TITLES.chat;
  }

  requestAnimationFrame(() => {
    syncCanvasSizesAndDraw();
  });
}

document.querySelectorAll(".sidebar-nav .nav-item[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setWorkspaceMode(btn.dataset.view, false));
});

btnSplitView.addEventListener("click", () => {
  setWorkspaceMode(currentView, !isSplitMode);
});

btnToggleSidebar.addEventListener("click", () => {
  appSidebar.classList.toggle("collapsed");
  localStorage.setItem("nz3_sidebar_collapsed", appSidebar.classList.contains("collapsed") ? "1" : "0");
  setTimeout(syncCanvasSizesAndDraw, 220);
});

if (localStorage.getItem("nz3_sidebar_collapsed") === "1") {
  appSidebar.classList.add("collapsed");
}

window.addEventListener("keydown", (e) => {
  if (e.altKey && !e.ctrlKey && !e.metaKey) {
    if (e.key === "1") {
      e.preventDefault();
      setWorkspaceMode("chat", false);
    } else if (e.key === "2") {
      e.preventDefault();
      setWorkspaceMode("inspector", false);
    } else if (e.key === "3") {
      e.preventDefault();
      setWorkspaceMode("knowledge", false);
    } else if (e.key.toLowerCase() === "s") {
      e.preventDefault();
      setWorkspaceMode(currentView, !isSplitMode);
    }
  }
});

// DIVISOR REDIMENSIONABLE EN SPLIT VIEW
let isResizingSplit = false;

splitResizer.addEventListener("pointerdown", (e) => {
  if (!isSplitMode) return;
  isResizingSplit = true;
  splitResizer.classList.add("dragging");
  splitResizer.setPointerCapture(e.pointerId);
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
});

splitResizer.addEventListener("pointermove", (e) => {
  if (!isResizingSplit) return;
  const rect = mainWorkspace.getBoundingClientRect();
  if (rect.width <= 0) return;
  const relPct = ((e.clientX - rect.left) / rect.width) * 100;
  const clamped = Math.max(30, Math.min(70, relPct));
  document.documentElement.style.setProperty("--split-left", `${clamped.toFixed(2)}%`);
});

function stopSplitResize(e) {
  if (!isResizingSplit) return;
  isResizingSplit = false;
  splitResizer.classList.remove("dragging");
  try {
    splitResizer.releasePointerCapture(e.pointerId);
  } catch (_) {}
  document.body.style.cursor = "";
  document.body.style.userSelect = "";
  syncCanvasSizesAndDraw();
}

splitResizer.addEventListener("pointerup", stopSplitResize);
splitResizer.addEventListener("pointercancel", stopSplitResize);

// =========================================================================
// MENSAJES DEL CHAT CON ANIMACIÓN Y ACCIONES RLHF CON ÍCONOS SVG
// =========================================================================
function appendMessage(role, text, meta = {}) {
  const row = document.createElement("div");
  const extraCls = meta.mode === "logical_contradiction" ? " contradiction" : "";
  row.className = `msg-row ${role}${extraCls}`;

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  if (role === "ai") {
    bubble.innerHTML = formatRichText(text);
  } else {
    bubble.textContent = stripPictographicEmojis(text).trim();
  }
  row.appendChild(bubble);

  if (role === "ai") {
    if (meta.auto_learned_events && meta.auto_learned_events.length > 0) {
      const autoBox = document.createElement("div");
      autoBox.className = "auto-learned-box";
      const itemsHtml = meta.auto_learned_events
        .map((ev) => `<li>${escapeHtml(stripPictographicEmojis(ev.text))}</li>`)
        .join("");
      autoBox.innerHTML = `<strong>Auto-enseñanza consolidada (+${meta.auto_learned_events.length})</strong><ul>${itemsHtml}</ul>`;
      row.appendChild(autoBox);
    }

    const metaBar = document.createElement("div");
    metaBar.className = "msg-meta";

    if (meta.mode) {
      const modeSpan = document.createElement("span");
      modeSpan.className = "badge-mode";
      modeSpan.textContent = meta.tool_used
        ? `TOOL · ${stripPictographicEmojis(meta.tool_used).trim().toUpperCase()}`
        : MODE_LABELS[meta.mode] || String(meta.mode).toUpperCase();
      metaBar.appendChild(modeSpan);
    }

    if (typeof meta.confidence === "number") {
      const pct = Math.round(meta.confidence * 100);
      const confSpan = document.createElement("span");
      confSpan.className =
        pct >= 70 ? "badge-conf-high" : pct >= 42 ? "badge-conf-med" : "badge-conf-low";
      confSpan.textContent = `CONF ${pct}%`;
      metaBar.appendChild(confSpan);
    }

    if (typeof meta.loss === "number") {
      const lossSpan = document.createElement("span");
      lossSpan.textContent = `LOSS ${meta.loss.toFixed(4)}`;
      metaBar.appendChild(lossSpan);
    }

    if (!["curiosity_ask", "curiosity_cancelled", "agent_tool", "logical_contradiction", "backup_restore"].includes(meta.mode)) {
      const rlhfDiv = document.createElement("div");
      rlhfDiv.className = "rlhf-buttons";

      if (meta.episode_id) {
        const btnUp = document.createElement("button");
        btnUp.type = "button";
        btnUp.className = "rlhf-btn";
        btnUp.innerHTML = SVG_ICONS.thumbUp;
        btnUp.title = "Reforzar positivamente (+Reward)";
        btnUp.onclick = () => sendFeedback(meta.episode_id, "reward", null, btnUp);

        const btnDown = document.createElement("button");
        btnDown.type = "button";
        btnDown.className = "rlhf-btn";
        btnDown.innerHTML = SVG_ICONS.thumbDown;
        btnDown.title = "Penalizar asociación (-Reward)";
        btnDown.onclick = () => sendFeedback(meta.episode_id, "punish", null, btnDown);

        rlhfDiv.appendChild(btnUp);
        rlhfDiv.appendChild(btnDown);
      }

      const btnEdit = document.createElement("button");
      btnEdit.type = "button";
      btnEdit.className = "rlhf-btn";
      btnEdit.innerHTML = `${SVG_ICONS.edit}<span>Corregir</span>`;
      btnEdit.onclick = () => openCorrectionModal(meta.episode_id || "");
      rlhfDiv.appendChild(btnEdit);

      metaBar.appendChild(rlhfDiv);
    }

    row.appendChild(metaBar);
  }

  chatMessages.appendChild(row);
  const streamWrap = chatMessages.parentElement;
  if (streamWrap) {
    streamWrap.scrollTop = streamWrap.scrollHeight;
  }
}

// SELECTOR DE LAS 4 CABEZAS DE SELF-ATTENTION
headSwitcher.querySelectorAll(".head-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    headSwitcher.querySelectorAll(".head-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedAttentionHead = btn.dataset.head;
    renderTokenAttention();
  });
});

// EXPORTAR E IMPORTAR RESPALDO COMPLETO DEL CEREBRO
btnExportBackup.addEventListener("click", async () => {
  btnExportBackup.disabled = true;
  setSystemStatus(true, "EXPORTING SNAPSHOT");
  try {
    const res = await fetch("/api/export_backup");
    if (!res.ok) throw new Error("Error generando respaldo");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    a.href = url;
    a.download = `neurozero_v3_backup_${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    appendMessage(
      "ai",
      "**Respaldo completo exportado.** El archivo `.json` incluye pesos MHSA (`.npz`), vocabulario, memoria multi-turno, reglas lógicas, caché RAG y notas.",
      { mode: "backup_restore", confidence: 1.0 }
    );
  } catch (_) {
    appendMessage("ai", "No se pudo generar el respaldo del cerebro.", { confidence: 0 });
  } finally {
    btnExportBackup.disabled = false;
    setSystemStatus(false);
  }
});

btnImportBackup.addEventListener("click", () => backupUploadInput.click());

backupUploadInput.addEventListener("change", () => {
  const file = backupUploadInput.files[0];
  if (!file) return;
  setSystemStatus(true, "RESTORING SNAPSHOT");
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const parsed = JSON.parse(String(reader.result || "{}"));
      const res = await fetch("/api/import_backup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backup: parsed }),
      });
      const data = await res.json();
      if (data.ok) {
        if (data.brain_state) updateBrainUI(data.brain_state);
        appendMessage(
          "ai",
          `**${data.message}** Se restauraron todos los tensores MHSA, recuerdos episódicos, reglas simbólicas y documentos RAG.`,
          { mode: "backup_restore", confidence: 1.0 }
        );
      } else {
        appendMessage("ai", data.message || "Archivo de respaldo inválido.", { confidence: 0 });
      }
    } catch (_) {
      appendMessage("ai", "El archivo seleccionado no es un JSON de respaldo válido.", { confidence: 0 });
    } finally {
      backupUploadInput.value = "";
      setSystemStatus(false);
    }
  };
  reader.readAsText(file, "utf-8");
});

// SUBIDA DE DOCUMENTOS Y DATASETS CSV
btnAttachDoc.addEventListener("click", () => fileUploadInput.click());
if (btnUploadFromKnowledge) {
  btnUploadFromKnowledge.addEventListener("click", () => fileUploadInput.click());
}

fileUploadInput.addEventListener("change", () => {
  const file = fileUploadInput.files[0];
  if (!file) return;

  setSystemStatus(true, "INDEXING RAG · NUMPY");
  appendMessage("user", `Subiendo archivo para indexación / análisis: ${file.name}`);
  const reader = new FileReader();
  reader.onload = async () => {
    const dataUrl = String(reader.result || "");
    const base64Part = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
    try {
      const res = await fetch("/api/upload_doc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content_base64: base64Part }),
      });
      const data = await res.json();
      if (data.ok) {
        let msgBody =
          `**Archivo «${data.title}» indexado en caché vectorial RAG.**\n` +
          `• Fragmentos vectoriales en SQLite: **${data.chunks_added}**\n` +
          `• Axiomas extraídos al grafo: **+${data.facts_extracted}**`;
        if (data.csv_report) {
          msgBody += `\n\n${data.csv_report}`;
        } else {
          msgBody += `\n• Extracto inicial: *${data.summary}...*`;
        }
        appendMessage("ai", msgBody, { mode: "document_rag", confidence: 0.99, loss: data.loss });
        if (data.brain_state) updateBrainUI(data.brain_state);
      } else {
        appendMessage("ai", data.message || "Error procesando archivo.", { confidence: 0 });
      }
    } catch (_) {
      appendMessage("ai", "Error subiendo documento al servidor local.", { confidence: 0 });
    } finally {
      fileUploadInput.value = "";
      setSystemStatus(false);
    }
  };
  reader.readAsDataURL(file);
});

// INVESTIGACIÓN EN WIKIPEDIA
wikiResearchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = wikiQueryInput.value.trim();
  if (!q) return;
  btnWikiGo.disabled = true;
  setSystemStatus(true, "QUERYING WIKIPEDIA · RAG");
  appendMessage("user", `Investiga sobre: ${q}`);
  wikiQueryInput.value = "";

  try {
    const res = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    const data = await res.json();
    if (data.ok) {
      appendMessage(
        "ai",
        `**Estudio completado en Wikipedia: «${data.title}»**\n\n${data.summary}\n\n` +
          `*Indexados ${data.chunks_added} fragmentos en caché vectorial RAG y +${data.facts_extracted} hechos en el grafo.*`,
        { mode: "web_research", confidence: 0.98, episode_id: data.episode_id, loss: data.loss }
      );
      if (data.brain_state) updateBrainUI(data.brain_state);
    } else {
      appendMessage("ai", data.message || "No se encontró el artículo.", { mode: "web_research", confidence: 0.3 });
    }
  } finally {
    btnWikiGo.disabled = false;
    setSystemStatus(false);
  }
});

// TAREAS Y NOTAS RÁPIDAS
quickTaskForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const desc = quickTaskInput.value.trim();
  if (!desc) return;
  quickTaskInput.value = "";
  const res = await fetch("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "add", description: desc }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
});

quickNoteForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const content = quickNoteInput.value.trim();
  if (!content) return;
  quickNoteInput.value = "";
  const res = await fetch("/api/notes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "add", content }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
});

async function sendFeedback(episodeId, type, correctedResponse = null, btnEl = null) {
  setSystemStatus(true, "RLHF · UPDATING WEIGHTS");
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episode_id: episodeId || null,
        type,
        corrected_response: correctedResponse,
      }),
    });
    const data = await res.json();
    if (data.brain_state) updateBrainUI(data.brain_state);
    if (btnEl && data.ok) {
      const prevHtml = btnEl.innerHTML;
      btnEl.innerHTML = SVG_ICONS.check;
      setTimeout(() => (btnEl.innerHTML = prevHtml), 1200);
    }
    if (type === "correct" && data.ok) {
      appendMessage("ai", `Corrección entrenada: ${data.message}`, {
        mode: "chat_correction",
        confidence: 0.99,
        episode_id: data.episode ? data.episode.id : null,
        loss: data.loss,
      });
    }
  } finally {
    setSystemStatus(false);
  }
}

function openCorrectionModal(episodeId) {
  modalEpisodeId.value = episodeId || "";
  modalCorrectedText.value = "";
  correctionModal.classList.remove("hidden");
  modalCorrectedText.focus();
}

btnCancelModal.addEventListener("click", () => correctionModal.classList.add("hidden"));

correctionForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const epId = modalEpisodeId.value ? parseInt(modalEpisodeId.value, 10) : null;
  const corrected = modalCorrectedText.value.trim();
  if (!corrected) return;
  correctionModal.classList.add("hidden");
  await sendFeedback(epId, "correct", corrected);
});

// ENVIAR MENSAJE EN EL CHAT
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  const isWikiCmd = /^(investiga|busca\s+en\s+wikipedia|estudia\s+sobre)\b/i.test(text);
  setSystemStatus(true, isWikiCmd ? "QUERYING WIKIPEDIA · RAG" : "INFERENCE · MHSA FORWARD");
  appendMessage("user", text);
  chatInput.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    if (data.ok) {
      appendMessage("ai", data.reply, {
        mode: data.mode,
        tool_used: data.tool_used,
        confidence: data.confidence,
        episode_id: data.episode_id,
        loss: data.loss,
        auto_learned_events: data.auto_learned_events || [],
      });
      if (data.brain_state) updateBrainUI(data.brain_state);
    }
  } catch (_) {
    appendMessage("ai", "Error de conexión con el servidor local.", { confidence: 0 });
  } finally {
    setSystemStatus(false);
  }
});

document.querySelectorAll(".prompt-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    chatInput.value = chip.dataset.sample;
    chatInput.focus();
  });
});

// INTERRUPTORES DE CONFIGURACIÓN
autoWebToggle.addEventListener("change", async () => {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ auto_web_research: autoWebToggle.checked }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
});

autoSelfPlayToggle.addEventListener("change", async () => {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ auto_self_play: autoSelfPlayToggle.checked }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
});

curiosityToggle.addEventListener("change", async () => {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ curiosity_mode: curiosityToggle.checked }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
});

async function triggerSelfPlayStep() {
  if (systemStatusPill.dataset.state === "busy") return;
  setSystemStatus(true, "SELF-PLAY · REPLAY");
  try {
    const res = await fetch("/api/self_teach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.ok && data.brain_state) updateBrainUI(data.brain_state);
  } catch (_) {
  } finally {
    setSystemStatus(false);
  }
}

btnManualSelfPlay.addEventListener("click", triggerSelfPlayStep);

setInterval(() => {
  if (autoSelfPlayToggle && autoSelfPlayToggle.checked) triggerSelfPlayStep();
}, 9000);

btnTrainEpochs.addEventListener("click", async () => {
  btnTrainEpochs.disabled = true;
  setSystemStatus(true, "TRAINING +20 EPOCHS · ADAM");
  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 20 }),
    });
    const data = await res.json();
    if (data.brain_state) {
      updateBrainUI(data.brain_state);
      appendMessage(
        "ai",
        `**+20 épocas entrenadas en el Micro-Transformer MHSA (4 cabezas + LayerNorm + Context Backprop).** Loss convergente: \`${data.final_loss.toFixed(5)}\`.`,
        { mode: "neural_hybrid", confidence: 0.99, loss: data.final_loss }
      );
    }
  } finally {
    btnTrainEpochs.disabled = false;
    setSystemStatus(false);
  }
});

btnResetMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  resetMenu.classList.toggle("open");
});
document.addEventListener("click", () => resetMenu.classList.remove("open"));

btnResetSeed.addEventListener("click", async () => {
  if (!confirm("¿Reiniciar cerebro NeuroZero 3.0 y biblioteca RAG con la semilla inicial?")) return;
  setSystemStatus(true, "RESETTING ENGINE");
  try {
    const res = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_seed: true }),
    });
    const data = await res.json();
    if (data.brain_state) updateBrainUI(data.brain_state);
    chatMessages.innerHTML = "";
    appendMessage("ai", data.message, { mode: "neural_hybrid", confidence: 1.0 });
  } finally {
    setSystemStatus(false);
  }
});

btnResetBlank.addEventListener("click", async () => {
  if (!confirm("¿Dejar todo 100% en blanco (Tabula Rasa)?")) return;
  setSystemStatus(true, "RESETTING TABULA RASA");
  try {
    const res = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_seed: false }),
    });
    const data = await res.json();
    if (data.brain_state) updateBrainUI(data.brain_state);
    chatMessages.innerHTML = "";
    appendMessage("ai", "**Tabula Rasa v3 activada.** Pesos MHSA, memoria y documentos en cero absoluto.", {
      mode: "auto_self_taught",
      confidence: 1.0,
    });
  } finally {
    setSystemStatus(false);
  }
});

directTeachForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const stim = document.getElementById("teachStimulus").value.trim();
  const resp = document.getElementById("teachResponse").value.trim();
  if (!stim || !resp) return;
  setSystemStatus(true, "TRAINING ASSOCIATION");
  try {
    const res = await fetch("/api/teach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stimulus: stim, response: resp }),
    });
    const data = await res.json();
    if (data.ok) {
      document.getElementById("teachStimulus").value = "";
      document.getElementById("teachResponse").value = "";
      if (data.brain_state) updateBrainUI(data.brain_state);
    }
  } finally {
    setSystemStatus(false);
  }
});

if (factsFilterInput) {
  factsFilterInput.addEventListener("input", () => {
    renderFactsAndRules(cachedFacts, cachedRules);
  });
}

// =========================================================================
// ACTUALIZACIÓN DE TELEMETRÍA Y RENDERIZADO DE MÓDULOS
// =========================================================================
function updateBrainUI(state) {
  const m = state.metrics;
  const lossFormatted = Number(m.last_loss || 0).toFixed(4);

  metricParams.textContent = m.total_parameters.toLocaleString();
  metricVocab.textContent = m.vocab_size.toLocaleString();
  metricRagTasks.textContent = `${m.rag_docs_count || 0} / ${m.tasks_pending || 0}`;
  const gNorm = typeof m.grad_norm === "number" ? m.grad_norm.toFixed(2) : "0.00";
  metricSteps.textContent = `${m.steps_trained.toLocaleString()} (${gNorm})`;
  metricLoss.textContent = lossFormatted;

  sideMetricParams.textContent = m.total_parameters.toLocaleString();
  sideMetricLoss.textContent = lossFormatted;

  epochsLabel.textContent = `${m.epochs_trained} EPOCHS`;

  if (state.architecture) {
    const a = state.architecture;
    const hCount = a.num_heads || 4;
    archLabel.textContent = `${hCount}H·${a.input_dim} → ${a.hidden1_dim} → ${a.hidden2_dim} → ${a.output_dim}`;
  }

  curiosityToggle.checked = !!m.curiosity_mode;
  autoSelfPlayToggle.checked = m.auto_self_play !== false;
  autoWebToggle.checked = m.auto_web_research !== false;

  if (m.awaiting_teaching_for) {
    teachingBanner.classList.remove("hidden");
    pendingStimulusText.textContent = `«${m.awaiting_teaching_for}»`;
  } else {
    teachingBanner.classList.add("hidden");
  }

  latestHeadAttention = state.head_attention || {};
  latestFallbackAttention = state.token_attention || [];
  latestActivations = state.activations || null;
  latestLossHistory = state.loss_history || [];
  cachedFacts = state.facts || [];
  cachedRules = state.logic_rules || [];

  renderTokenAttention();
  renderWorkingMemory(state.working_memory || []);
  renderRagDocs(state.rag_docs || []);
  renderKanbanAndNotes(state.tasks || [], state.notes || []);
  renderFactsAndRules(cachedFacts, cachedRules);
  renderSelfTeachLog(state.self_teaching_log || []);
  renderEpisodesList(state.episodes || []);
  renderVocabCloud(state.top_vocab || []);
  drawNeuralNetwork(latestActivations);
  drawLossCurve(latestLossHistory);
}

function renderTokenAttention() {
  const attnList =
    (latestHeadAttention && latestHeadAttention[selectedAttentionHead]) ||
    latestFallbackAttention ||
    [];

  if (!attnList || attnList.length === 0) return;

  attnTokens.innerHTML = "";
  const maxW = Math.max(...attnList.map((a) => a.weight), 0.01);
  const headName = HEAD_LABELS[selectedAttentionHead] || "MHSA";

  attnList.forEach((item) => {
    const intensity = Math.min(1, item.weight / maxW);
    const chip = document.createElement("span");
    chip.className = "attn-chip";
    chip.style.backgroundColor = `rgba(16, 185, 129, ${(0.05 + intensity * 0.34).toFixed(3)})`;
    chip.style.borderColor = `rgba(16, 185, 129, ${(0.14 + intensity * 0.65).toFixed(3)})`;
    if (intensity > 0.72) {
      chip.style.boxShadow = "0 0 12px rgba(16, 185, 129, 0.16)";
    }
    chip.title = `${headName} · Peso: ${(item.weight * 100).toFixed(2)}%`;
    chip.innerHTML = `<span>${escapeHtml(item.token)}</span><span class="attn-chip-pct">${Math.round(
      item.weight * 100
    )}%</span>`;
    attnTokens.appendChild(chip);
  });
}

function renderWorkingMemory(turns) {
  if (!workingMemBox) return;
  if (!turns || turns.length === 0) {
    workingMemBox.innerHTML = `<span class="empty-hint">Buffer circular multi-turno (máx. 4 turnos · decaimiento exponencial): sin turnos activos.</span>`;
    return;
  }
  const chips = turns
    .map(
      (t) =>
        `<span class="wm-chip" title="${escapeHtml(t.turn)}">w=${t.weight.toFixed(2)} · ${escapeHtml(
          t.turn.slice(0, 38)
        )}${t.turn.length > 38 ? "…" : ""}</span>`
    )
    .join("");
  workingMemBox.innerHTML = `<span class="subbar-label">Buffer Circular Activo (4 turnos · decaimiento exponencial)</span><div class="wm-chips">${chips}</div>`;
}

function renderRagDocs(docs) {
  ragDocsList.innerHTML = "";
  if (docs.length === 0) {
    ragDocsList.innerHTML = `<span class="empty-hint">Sin documentos en caché vectorial. Sube un archivo o investiga un artículo en Wikipedia.</span>`;
    return;
  }
  docs.forEach((d) => {
    const isWiki = d.source_type === "wikipedia";
    const isCsv = d.title.toLowerCase().endsWith(".csv");
    const svgIcon = isWiki ? SVG_ICONS.globe : isCsv ? SVG_ICONS.table : SVG_ICONS.file;
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">${svgIcon}<span>${escapeHtml(d.title)}</span></span>
        <span class="resp-tag mono">${d.chunks_count} chunks · ${escapeHtml(d.date)}</span>
      </div>
      <div class="item-badges">
        <button type="button" class="btn-del" title="Eliminar documento">${SVG_ICONS.close}</button>
      </div>
    `;
    item.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/delete_doc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: d.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    ragDocsList.appendChild(item);
  });
}

function renderKanbanAndNotes(tasks, notes) {
  const pending = tasks.filter((t) => !t.done);
  const done = tasks.filter((t) => t.done);

  kanbanPendingCount.textContent = String(pending.length);
  kanbanDoneCount.textContent = String(done.length);
  tasksCountBadge.textContent = `${pending.length} PENDING`;
  notesCountBadge.textContent = String(notes.length);

  kanbanPendingList.innerHTML = "";
  kanbanDoneList.innerHTML = "";
  notesBoardList.innerHTML = "";

  if (pending.length === 0) {
    kanbanPendingList.innerHTML = `<span class="empty-hint">Sin pendientes</span>`;
  }
  if (done.length === 0) {
    kanbanDoneList.innerHTML = `<span class="empty-hint">Sin completadas</span>`;
  }

  const createCard = (t) => {
    const card = document.createElement("div");
    card.className = "kanban-card";
    card.innerHTML = `
      <label class="kanban-check-label">
        <input type="checkbox" ${t.done ? "checked" : ""} />
        <span class="${t.done ? "task-done-text" : ""}">
          <span class="mono accent-text">#${t.id}</span> ${escapeHtml(t.description)}
        </span>
      </label>
      <button type="button" class="btn-del" title="Eliminar tarea">${SVG_ICONS.close}</button>
    `;
    card.querySelector("input").addEventListener("change", async (e) => {
      const res = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "toggle", id: t.id, done: e.target.checked }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    card.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "delete", id: t.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    return card;
  };

  pending.forEach((t) => kanbanPendingList.appendChild(createCard(t)));
  done.forEach((t) => kanbanDoneList.appendChild(createCard(t)));

  if (notes.length === 0) {
    notesBoardList.innerHTML = `<span class="empty-hint">Sin notas guardadas.</span>`;
    return;
  }

  notes.forEach((n) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">${SVG_ICONS.note}<span class="mono">NOTA #${n.id} · ${escapeHtml(n.date)}</span></span>
        <span class="resp-tag">${escapeHtml(n.content)}</span>
      </div>
      <button type="button" class="btn-del" title="Eliminar nota">${SVG_ICONS.close}</button>
    `;
    item.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "delete", id: n.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    notesBoardList.appendChild(item);
  });
}

function renderFactsAndRules(facts, rules) {
  factsList.innerHTML = "";
  const totalCount = facts.length + rules.length;
  factsCountBadge.textContent = `${totalCount} AXIOMS`;

  const q = (factsFilterInput ? factsFilterInput.value : "").trim().toLowerCase();

  const filteredRules = rules.filter(
    (r) => !q || String(r.condition).toLowerCase().includes(q) || String(r.consequence).toLowerCase().includes(q)
  );
  const filteredFacts = facts.filter(
    (f) =>
      !q ||
      String(f.subject).toLowerCase().includes(q) ||
      String(f.relation).toLowerCase().includes(q) ||
      String(f.object).toLowerCase().includes(q)
  );

  if (filteredRules.length === 0 && filteredFacts.length === 0) {
    factsList.innerHTML = `<span class="empty-hint">No hay hechos ni reglas que coincidan con el filtro.</span>`;
    return;
  }

  filteredRules.forEach((r) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">${SVG_ICONS.rule}<span class="mono">REGLA #${r.id}</span></span>
        <span class="resp-tag">Si <strong>${escapeHtml(r.condition)}</strong> → entonces <strong>${escapeHtml(
      r.consequence
    )}</strong></span>
      </div>
    `;
    factsList.appendChild(item);
  });

  filteredFacts.forEach((f) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">${SVG_ICONS.node}<span>${escapeHtml(f.subject)} <span class="mono accent-text">[${escapeHtml(
      f.relation
    )}]</span></span></span>
        <span class="resp-tag">${escapeHtml(f.object)}</span>
      </div>
      <button type="button" class="btn-del" title="Eliminar axioma">${SVG_ICONS.close}</button>
    `;
    item.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/delete_fact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: f.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    factsList.appendChild(item);
  });
}

function renderSelfTeachLog(logs) {
  selfTeachList.innerHTML = "";
  logs.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "list-item";
    const lossTag =
      typeof entry.loss === "number" ? `<span class="reward-pill">${entry.loss.toFixed(4)}</span>` : "";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag"><span class="mono accent-text">${escapeHtml(entry.time)}</span><span>${escapeHtml(
      stripPictographicEmojis(entry.summary)
    )}</span></span>
        <span class="resp-tag mono">${escapeHtml(stripPictographicEmojis(entry.detail))}</span>
      </div>
      <div class="item-badges">${lossTag}</div>
    `;
    selfTeachList.appendChild(item);
  });
}

function renderEpisodesList(episodes) {
  episodesList.innerHTML = "";
  episodes.forEach((ep) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag"><span>${escapeHtml(stripPictographicEmojis(ep.stimulus))}</span></span>
        <span class="resp-tag">${escapeHtml(stripPictographicEmojis(ep.response))}</span>
      </div>
      <div class="item-badges">
        <span class="reward-pill">R=${Number(ep.reward).toFixed(2)}</span>
        <button type="button" class="btn-del" title="Eliminar recuerdo">${SVG_ICONS.close}</button>
      </div>
    `;
    item.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/delete_episode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ep.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    episodesList.appendChild(item);
  });
}

function renderVocabCloud(vocab) {
  vocabCloud.innerHTML = "";
  vocab.slice(0, 32).forEach((v) => {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    chip.textContent = `${v.word} · ${v.freq}`;
    chip.addEventListener("click", async () => {
      const res = await fetch("/api/neighbors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: v.word }),
      });
      const data = await res.json();
      neighborsResult.classList.remove("hidden");
      if (data.neighbors && data.neighbors.length > 0) {
        const list = data.neighbors
          .map((n) => `<strong class="mono accent-text">${escapeHtml(n.word)}</strong> <span class="mono">(${n.similarity})</span>`)
          .join(", ");
        neighborsResult.innerHTML = `Vecinos coseno de <code class="mono">${escapeHtml(v.word)}</code> en <code class="mono">W_emb</code>: ${list}`;
      } else {
        neighborsResult.innerHTML = `Sin vecinos cercanos aún para <code class="mono">${escapeHtml(v.word)}</code>.`;
      }
    });
    vocabCloud.appendChild(chip);
  });
}

// =========================================================================
// HIGH-DPI (RETINA) CANVAS ENGINE + RESIZEOBSERVER + BÉZIER + GLOW + TOOLTIPS
// =========================================================================
function resizeHiDPICanvas(viewportEl, canvasEl, ctx) {
  if (!viewportEl || !canvasEl) return null;
  const rect = viewportEl.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return null;

  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);

  if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
    canvasEl.width = targetW;
    canvasEl.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: rect.width, h: rect.height };
}

function syncCanvasSizesAndDraw() {
  const s1 = resizeHiDPICanvas(nnViewport, nnCanvas, nnCtx);
  if (s1) {
    nnSize = s1;
    drawNeuralNetwork(latestActivations);
  }
  const s2 = resizeHiDPICanvas(lossViewport, lossCanvas, lossCtx);
  if (s2) {
    lossSize = s2;
    drawLossCurve(latestLossHistory);
  }
}

const canvasResizeObserver = new ResizeObserver(() => {
  syncCanvasSizesAndDraw();
});
if (nnViewport) canvasResizeObserver.observe(nnViewport);
if (lossViewport) canvasResizeObserver.observe(lossViewport);

function drawSubtleDotGrid(ctx, w, h) {
  ctx.fillStyle = "rgba(255, 255, 255, 0.03)";
  const step = 22;
  for (let x = 14; x < w; x += step) {
    for (let y = 14; y < h - 20; y += step) {
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

function drawNeuralNetwork(activations) {
  const { w, h } = nnSize;
  if (w < 20 || h < 20) return;
  nnCtx.clearRect(0, 0, w, h);
  drawSubtleDotGrid(nnCtx, w, h);

  const layers = [
    { name: "MHSA (4x16)", vals: (activations && activations.input) || new Array(8).fill(0.08), count: 8 },
    { name: "Swish FFN", vals: (activations && activations.hidden1) || new Array(10).fill(0.08), count: 10 },
    { name: "Res + LN", vals: (activations && activations.hidden2) || new Array(8).fill(0.08), count: 8 },
    { name: "Latent L2", vals: (activations && activations.output) || new Array(8).fill(0.08), count: 8 },
  ];

  const padX = Math.max(42, w * 0.1);
  const usableW = Math.max(80, w - padX * 2);
  const topPad = 14;
  const bottomPad = 26;
  const usableH = Math.max(60, h - topPad - bottomPad);

  const positions = [];
  nnHitNodes = [];

  layers.forEach((layer, lIdx) => {
    const x = padX + (usableW * lIdx) / (layers.length - 1);
    const nodes = [];
    const n = layer.count;
    for (let i = 0; i < n; i++) {
      const y = topPad + (usableH * (i + 0.5)) / n;
      const val = Number(layer.vals[i % layer.vals.length] || 0);
      const nodeObj = { layerIdx: lIdx, layerName: layer.name, index: i, x, y, val };
      nodes.push(nodeObj);
      nnHitNodes.push(nodeObj);
    }
    positions.push({ name: layer.name, x, nodes });
  });

  // Curvas Bézier cúbicas suaves entre capas
  for (let l = 0; l < positions.length - 1; l++) {
    const curr = positions[l].nodes;
    const next = positions[l + 1].nodes;
    for (let i = 0; i < curr.length; i++) {
      const n1 = curr[i];
      for (let j = 0; j < next.length; j++) {
        const n2 = next[j];
        const prod = n1.val * n2.val;
        const mag = Math.min(1, Math.abs(prod) * 2.4 + 0.05);
        const isHighlighted =
          hoveredNnNode &&
          ((hoveredNnNode.layerIdx === l && hoveredNnNode.index === i) ||
            (hoveredNnNode.layerIdx === l + 1 && hoveredNnNode.index === j));

        const cpOffset = (n2.x - n1.x) * 0.48;
        nnCtx.beginPath();
        nnCtx.moveTo(n1.x, n1.y);
        nnCtx.bezierCurveTo(n1.x + cpOffset, n1.y, n2.x - cpOffset, n2.y, n2.x, n2.y);

        if (isHighlighted) {
          nnCtx.strokeStyle = prod >= 0 ? "rgba(16, 185, 129, 0.85)" : "rgba(148, 163, 184, 0.7)";
          nnCtx.lineWidth = 1.65;
        } else {
          nnCtx.strokeStyle =
            prod >= 0
              ? `rgba(16, 185, 129, ${(mag * 0.44).toFixed(3)})`
              : `rgba(148, 163, 184, ${(mag * 0.24).toFixed(3)})`;
          nnCtx.lineWidth = 0.55 + mag * 1.1;
        }
        nnCtx.stroke();
      }
    }
  }

  // Neuronas con resplandor sutil (glow) en las activaciones intensas
  positions.forEach((layer) => {
    layer.nodes.forEach((node) => {
      const absVal = Math.min(1.5, Math.abs(node.val));
      const normMag = Math.min(1, absVal);
      const isHovered =
        hoveredNnNode && hoveredNnNode.layerIdx === node.layerIdx && hoveredNnNode.index === node.index;
      const radius = (isHovered ? 5.6 : 3.6) + normMag * 2.3;

      if (normMag > 0.32 || isHovered) {
        const glowRadius = radius * (isHovered ? 3.2 : 2.5);
        const grad = nnCtx.createRadialGradient(node.x, node.y, radius * 0.3, node.x, node.y, glowRadius);
        if (node.val >= 0) {
          grad.addColorStop(0, `rgba(16, 185, 129, ${(0.38 * normMag + (isHovered ? 0.25 : 0)).toFixed(3)})`);
          grad.addColorStop(1, "rgba(16, 185, 129, 0)");
        } else {
          grad.addColorStop(0, `rgba(148, 163, 184, ${(0.3 * normMag + (isHovered ? 0.2 : 0)).toFixed(3)})`);
          grad.addColorStop(1, "rgba(148, 163, 184, 0)");
        }
        nnCtx.fillStyle = grad;
        nnCtx.beginPath();
        nnCtx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        nnCtx.fill();
      }

      nnCtx.beginPath();
      nnCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      nnCtx.fillStyle =
        node.val >= 0
          ? `rgba(16, 185, 129, ${(0.35 + normMag * 0.65).toFixed(3)})`
          : `rgba(100, 116, 139, ${(0.45 + normMag * 0.5).toFixed(3)})`;
      nnCtx.fill();

      nnCtx.strokeStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.45)";
      nnCtx.lineWidth = isHovered ? 1.5 : 0.8;
      nnCtx.stroke();
    });

    nnCtx.fillStyle = "#646b78";
    nnCtx.font = "500 10px 'JetBrains Mono', monospace";
    nnCtx.textAlign = "center";
    nnCtx.fillText(layer.name, layer.x, h - 8);
  });
}

function drawLossCurve(lossHistory) {
  const { w, h } = lossSize;
  if (w < 20 || h < 20) return;
  lossCtx.clearRect(0, 0, w, h);

  const padLeft = 46;
  const padRight = 16;
  const padTop = 18;
  const padBottom = 24;
  const plotW = Math.max(40, w - padLeft - padRight);
  const plotH = Math.max(40, h - padTop - padBottom);

  const maxLoss = lossHistory && lossHistory.length > 0 ? Math.max(...lossHistory, 0.25) : 1.0;

  // Líneas guía horizontales y etiquetas monoespaciadas
  lossCtx.font = "400 10px 'JetBrains Mono', monospace";
  lossCtx.textAlign = "right";
  for (let i = 0; i <= 3; i++) {
    const frac = i / 3;
    const y = padTop + plotH * frac;
    const labelVal = maxLoss * (1 - frac);

    lossCtx.strokeStyle = "rgba(255, 255, 255, 0.055)";
    lossCtx.lineWidth = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(padLeft, y);
    lossCtx.lineTo(w - padRight, y);
    lossCtx.stroke();

    lossCtx.fillStyle = "#646b78";
    lossCtx.fillText(labelVal.toFixed(2), padLeft - 7, y + 3);
  }

  lossHitPoints = [];
  if (!lossHistory || lossHistory.length === 0) return;

  const pts = lossHistory.map((val, idx) => {
    const x = padLeft + (lossHistory.length === 1 ? plotW / 2 : (idx / (lossHistory.length - 1)) * plotW);
    const normY = Math.min(1, Math.max(0, val / (maxLoss + 1e-6)));
    const y = padTop + plotH * (1 - normY);
    return { x, y, val, idx };
  });
  lossHitPoints = pts;

  // Área degradada bajo la curva Bézier suave
  const areaGrad = lossCtx.createLinearGradient(0, padTop, 0, padTop + plotH);
  areaGrad.addColorStop(0, "rgba(16, 185, 129, 0.24)");
  areaGrad.addColorStop(1, "rgba(16, 185, 129, 0.0)");

  lossCtx.beginPath();
  lossCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 0; i < pts.length - 1; i++) {
    const xc = (pts[i].x + pts[i + 1].x) / 2;
    const yc = (pts[i].y + pts[i + 1].y) / 2;
    lossCtx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
  }
  lossCtx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  lossCtx.lineTo(pts[pts.length - 1].x, padTop + plotH);
  lossCtx.lineTo(pts[0].x, padTop + plotH);
  lossCtx.closePath();
  lossCtx.fillStyle = areaGrad;
  lossCtx.fill();

  // Trazo principal suave
  lossCtx.beginPath();
  lossCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 0; i < pts.length - 1; i++) {
    const xc = (pts[i].x + pts[i + 1].x) / 2;
    const yc = (pts[i].y + pts[i + 1].y) / 2;
    lossCtx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
  }
  lossCtx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  lossCtx.strokeStyle = "#10b981";
  lossCtx.lineWidth = 2;
  lossCtx.stroke();

  // Punto final brillante
  const lastPt = pts[pts.length - 1];
  const endGlow = lossCtx.createRadialGradient(lastPt.x, lastPt.y, 1, lastPt.x, lastPt.y, 11);
  endGlow.addColorStop(0, "rgba(16, 185, 129, 0.65)");
  endGlow.addColorStop(1, "rgba(16, 185, 129, 0)");
  lossCtx.fillStyle = endGlow;
  lossCtx.beginPath();
  lossCtx.arc(lastPt.x, lastPt.y, 11, 0, Math.PI * 2);
  lossCtx.fill();

  lossCtx.beginPath();
  lossCtx.arc(lastPt.x, lastPt.y, 3.3, 0, Math.PI * 2);
  lossCtx.fillStyle = "#10b981";
  lossCtx.fill();

  // Crosshair de inspección al pasar el cursor
  if (hoveredLossIdx !== null && pts[hoveredLossIdx]) {
    const hp = pts[hoveredLossIdx];
    lossCtx.strokeStyle = "rgba(16, 185, 129, 0.35)";
    lossCtx.lineWidth = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(hp.x, padTop);
    lossCtx.lineTo(hp.x, padTop + plotH);
    lossCtx.stroke();

    lossCtx.beginPath();
    lossCtx.arc(hp.x, hp.y, 4.5, 0, Math.PI * 2);
    lossCtx.fillStyle = "#10b981";
    lossCtx.fill();
    lossCtx.strokeStyle = "#ffffff";
    lossCtx.lineWidth = 1.4;
    lossCtx.stroke();
  }
}

function showCanvasTooltip(clientX, clientY, htmlContent) {
  canvasTooltip.innerHTML = htmlContent;
  canvasTooltip.classList.remove("hidden");
  const offset = 14;
  let left = clientX + offset;
  let top = clientY - 12;
  const rect = canvasTooltip.getBoundingClientRect();
  if (left + rect.width > window.innerWidth - 10) {
    left = clientX - rect.width - offset;
  }
  if (top + rect.height > window.innerHeight - 10) {
    top = window.innerHeight - rect.height - 10;
  }
  canvasTooltip.style.left = `${Math.max(8, left)}px`;
  canvasTooltip.style.top = `${Math.max(8, top)}px`;
}

function hideCanvasTooltip() {
  canvasTooltip.classList.add("hidden");
}

nnCanvas.addEventListener("mousemove", (e) => {
  const rect = nnCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let closest = null;
  let bestDist = 15;
  for (const node of nnHitNodes) {
    const d = Math.hypot(mx - node.x, my - node.y);
    if (d < bestDist) {
      bestDist = d;
      closest = node;
    }
  }

  if (closest !== hoveredNnNode) {
    hoveredNnNode = closest;
    drawNeuralNetwork(latestActivations);
  }

  if (closest) {
    const sign = closest.val >= 0 ? "+" : "";
    showCanvasTooltip(
      e.clientX,
      e.clientY,
      `<span class="accent-text">${escapeHtml(closest.layerName)}</span> · N#${String(closest.index + 1).padStart(
        2,
        "0"
      )}<br/>Activación: <strong>${sign}${closest.val.toFixed(4)}</strong>`
    );
  } else {
    hideCanvasTooltip();
  }
});

nnCanvas.addEventListener("mouseleave", () => {
  if (hoveredNnNode !== null) {
    hoveredNnNode = null;
    drawNeuralNetwork(latestActivations);
  }
  hideCanvasTooltip();
});

lossCanvas.addEventListener("mousemove", (e) => {
  if (!lossHitPoints || lossHitPoints.length === 0) return;
  const rect = lossCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;

  let bestIdx = 0;
  let bestDx = Infinity;
  lossHitPoints.forEach((pt, idx) => {
    const dx = Math.abs(mx - pt.x);
    if (dx < bestDx) {
      bestDx = dx;
      bestIdx = idx;
    }
  });

  if (bestDx <= 28) {
    if (hoveredLossIdx !== bestIdx) {
      hoveredLossIdx = bestIdx;
      drawLossCurve(latestLossHistory);
    }
    const pt = lossHitPoints[bestIdx];
    showCanvasTooltip(
      e.clientX,
      e.clientY,
      `Sample <span class="accent-text">#${pt.idx + 1}</span> · Loss: <strong>${pt.val.toFixed(5)}</strong>`
    );
  } else {
    if (hoveredLossIdx !== null) {
      hoveredLossIdx = null;
      drawLossCurve(latestLossHistory);
    }
    hideCanvasTooltip();
  }
});

lossCanvas.addEventListener("mouseleave", () => {
  if (hoveredLossIdx !== null) {
    hoveredLossIdx = null;
    drawLossCurve(latestLossHistory);
  }
  hideCanvasTooltip();
});

// =========================================================================
// INICIALIZACIÓN DEL SISTEMA
// =========================================================================
async function initApp() {
  try {
    setSystemStatus(true, "BOOTING ENGINE");
    const res = await fetch("/api/state");
    const data = await res.json();
    if (data.ok && data.brain_state) {
      updateBrainUI(data.brain_state);
      appendMessage(
        "ai",
        "**NeuroZero 3.0 Cyber-Lab activo.**\n" +
          "• **Micro-Transformer MHSA:** `4` cabezas de atención (`Q·Kᵀ·V`), *LayerNorm* analítico, retropropagación de contexto y *Global L2 Gradient Clipping*.\n" +
          "• **Memoria Multi-Turno & Reranker:** Buffer circular con decaimiento exponencial y *Neural-Markov Reranker* (`K = 5`) + caché vectorial RAG.\n" +
          "• **Razonamiento Simbólico:** Detección de contradicciones lógicas y árboles de demostración con *Backward Chaining* (`¿Por qué X es Y?`).\n" +
          "• **Interfaz Modular:** Alterna vistas desde la barra lateral (`Alt+1`, `Alt+2`, `Alt+3`) o activa **Split View** (`Alt+S`) para inspeccionar las neuronas en vivo mientras conversas.",
        { mode: "neural_hybrid", confidence: 1.0 }
      );
    }
  } catch (err) {
    console.error("Error inicializando NeuroZero 3.0:", err);
  } finally {
    setSystemStatus(false);
    syncCanvasSizesAndDraw();
  }
}

initApp();
