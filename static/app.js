// NeuroZero 2.0 Frontend — Micro-Transformer + RAG + Lógica + Agente

const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const btnAttachDoc = document.getElementById("btnAttachDoc");
const fileUploadInput = document.getElementById("fileUploadInput");
const teachingBanner = document.getElementById("teachingBanner");
const pendingStimulusText = document.getElementById("pendingStimulusText");

const autoWebToggle = document.getElementById("autoWebToggle");
const autoSelfPlayToggle = document.getElementById("autoSelfPlayToggle");
const curiosityToggle = document.getElementById("curiosityToggle");
const btnTrainEpochs = document.getElementById("btnTrainEpochs");
const btnManualSelfPlay = document.getElementById("btnManualSelfPlay");

const btnResetMenu = document.getElementById("btnResetMenu");
const resetMenu = document.getElementById("resetMenu");
const btnResetSeed = document.getElementById("btnResetSeed");
const btnResetBlank = document.getElementById("btnResetBlank");

// Métricas y Atención
const metricParams = document.getElementById("metricParams");
const metricVocab = document.getElementById("metricVocab");
const metricRagTasks = document.getElementById("metricRagTasks");
const metricSteps = document.getElementById("metricSteps");
const metricLoss = document.getElementById("metricLoss");
const epochsLabel = document.getElementById("epochsLabel");
const archLabel = document.getElementById("archLabel");
const attnTokens = document.getElementById("attnTokens");

// Pestañas y Contenedores
const ragDocsList = document.getElementById("ragDocsList");
const wikiResearchForm = document.getElementById("wikiResearchForm");
const wikiQueryInput = document.getElementById("wikiQueryInput");
const btnWikiGo = document.getElementById("btnWikiGo");

const quickTaskForm = document.getElementById("quickTaskForm");
const quickTaskInput = document.getElementById("quickTaskInput");
const quickNoteForm = document.getElementById("quickNoteForm");
const quickNoteInput = document.getElementById("quickNoteInput");
const agentTasksNotesList = document.getElementById("agentTasksNotesList");

const selfTeachList = document.getElementById("selfTeachList");
const episodesList = document.getElementById("episodesList");
const factsList = document.getElementById("factsList");
const vocabCloud = document.getElementById("vocabCloud");
const neighborsResult = document.getElementById("neighborsResult");
const directTeachForm = document.getElementById("directTeachForm");

// Modal Corrección
const correctionModal = document.getElementById("correctionModal");
const correctionForm = document.getElementById("correctionForm");
const modalEpisodeId = document.getElementById("modalEpisodeId");
const modalCorrectedText = document.getElementById("modalCorrectedText");
const btnCancelModal = document.getElementById("btnCancelModal");

// Canvas
const nnCanvas = document.getElementById("nnCanvas");
const nnCtx = nnCanvas.getContext("2d");
const lossCanvas = document.getElementById("lossCanvas");
const lossCtx = lossCanvas.getContext("2d");

const MODE_LABELS = {
  neural_hybrid: "⚡ Micro-Transformer",
  auto_self_taught: "🧠 Auto-Aprendido",
  autonomous_synthesis: "🔮 Síntesis Autónoma",
  knowledge_graph: "🌐 Grafo Semántico",
  logical_reasoning: "🧩 Inferencia Lógica",
  document_rag: "📚 Memoria RAG",
  web_research: "🌐 Auto-Investigación Wiki",
  agent_tool: "🛠️ Herramienta de Agente",
  curiosity_ask: "🤔 Pregunta Manual",
  learned_from_curiosity: "🧠 Aprendido en Chat",
  explicit_teaching: "🎯 Regla Enseñada",
  chat_correction: "🔧 Corregido en Vivo",
};

function formatRichText(raw) {
  let html = escapeHtml(raw);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  return html;
}

function appendMessage(role, text, meta = {}) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  if (role === "ai") {
    bubble.innerHTML = formatRichText(text);
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);

  if (role === "ai") {
    if (meta.auto_learned_events && meta.auto_learned_events.length > 0) {
      const autoBox = document.createElement("div");
      autoBox.className = "auto-learned-box";
      const itemsHtml = meta.auto_learned_events
        .map((ev) => `<li>${escapeHtml(ev.text)}</li>`)
        .join("");
      autoBox.innerHTML = `<strong>⚡ Auto-enseñanza extraída (+${meta.auto_learned_events.length}):</strong><ul>${itemsHtml}</ul>`;
      row.appendChild(autoBox);
    }

    const metaBar = document.createElement("div");
    metaBar.className = "msg-meta";

    if (meta.mode) {
      const modeSpan = document.createElement("span");
      modeSpan.className = "badge-mode";
      modeSpan.textContent = meta.tool_used
        ? `🛠️ ${meta.tool_used}`
        : MODE_LABELS[meta.mode] || meta.mode;
      metaBar.appendChild(modeSpan);
    }

    if (typeof meta.confidence === "number") {
      const pct = Math.round(meta.confidence * 100);
      const confSpan = document.createElement("span");
      confSpan.className =
        pct >= 70 ? "badge-conf-high" : pct >= 42 ? "badge-conf-med" : "badge-conf-low";
      confSpan.textContent = `Confianza: ${pct}%`;
      metaBar.appendChild(confSpan);
    }

    if (typeof meta.loss === "number") {
      const lossSpan = document.createElement("span");
      lossSpan.textContent = `Loss: ${meta.loss.toFixed(4)}`;
      metaBar.appendChild(lossSpan);
    }

    if (!["curiosity_ask", "curiosity_cancelled", "agent_tool"].includes(meta.mode)) {
      const rlhfDiv = document.createElement("div");
      rlhfDiv.className = "rlhf-buttons";

      if (meta.episode_id) {
        const btnUp = document.createElement("button");
        btnUp.className = "rlhf-btn";
        btnUp.textContent = "👍";
        btnUp.title = "Premiar respuesta";
        btnUp.onclick = () => sendFeedback(meta.episode_id, "reward", null, btnUp);

        const btnDown = document.createElement("button");
        btnDown.className = "rlhf-btn";
        btnDown.textContent = "👎";
        btnDown.title = "Castigar respuesta";
        btnDown.onclick = () => sendFeedback(meta.episode_id, "punish", null, btnDown);

        rlhfDiv.appendChild(btnUp);
        rlhfDiv.appendChild(btnDown);
      }

      const btnEdit = document.createElement("button");
      btnEdit.className = "rlhf-btn";
      btnEdit.textContent = "✏️ Corregir";
      btnEdit.onclick = () => openCorrectionModal(meta.episode_id || "");
      rlhfDiv.appendChild(btnEdit);

      metaBar.appendChild(rlhfDiv);
    }

    row.appendChild(metaBar);
  }

  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// SUBIDA DE DOCUMENTOS (RAG)
btnAttachDoc.addEventListener("click", () => fileUploadInput.click());

fileUploadInput.addEventListener("change", () => {
  const file = fileUploadInput.files[0];
  if (!file) return;

  appendMessage("user", `📎 Subiendo documento para estudio RAG: ${file.name}`);
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
        appendMessage(
          "ai",
          `📚 **¡Documento «${data.title}» estudiado e indexado!**\n` +
            `• Fragmentos vectoriales en SQLite: **${data.chunks_added}**\n` +
            `• Hechos semánticos extraídos al grafo: **+${data.facts_extracted}**\n` +
            `• Vista previa: *${data.summary}...*\n\n` +
            `¡Ya puedes hacerme preguntas sobre el contenido de este archivo!`,
          { mode: "document_rag", confidence: 0.99, loss: data.loss }
        );
        if (data.brain_state) updateBrainUI(data.brain_state);
      } else {
        appendMessage("ai", `⚠️ ${data.message || "Error procesando archivo."}`, { confidence: 0 });
      }
    } catch (err) {
      appendMessage("ai", "Error subiendo documento al motor local.", { confidence: 0 });
    }
    fileUploadInput.value = "";
  };
  reader.readAsDataURL(file);
});

// AUTO-INVESTIGACIÓN WIKIPEDIA DESDE EL PANEL DERECHO
wikiResearchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = wikiQueryInput.value.trim();
  if (!q) return;
  btnWikiGo.disabled = true;
  btnWikiGo.textContent = "⏳...";
  appendMessage("user", `🌐 Investiga sobre: ${q}`);
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
        `🌐 **Estudio completado en Wikipedia: «${data.title}»**\n\n${data.summary}\n\n` +
          `📚 *Guardé ${data.chunks_added} fragmentos en mi memoria RAG y extraje +${data.facts_extracted} hechos.*`,
        { mode: "web_research", confidence: 0.98, episode_id: data.episode_id, loss: data.loss }
      );
      if (data.brain_state) updateBrainUI(data.brain_state);
    } else {
      appendMessage("ai", `⚠️ ${data.message}`, { mode: "web_research", confidence: 0.3 });
    }
  } finally {
    btnWikiGo.disabled = false;
    btnWikiGo.textContent = "🌐 Estudiar";
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
    const orig = btnEl.textContent;
    btnEl.textContent = "✅";
    setTimeout(() => (btnEl.textContent = orig), 1300);
  }
  if (type === "correct" && data.ok) {
    appendMessage("ai", `🔧 Corrección entrenada: ${data.message}`, {
      mode: "chat_correction",
      confidence: 0.99,
      episode_id: data.episode ? data.episode.id : null,
      loss: data.loss,
    });
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
  } catch (err) {
    appendMessage("ai", "Error de conexión con el servidor local.", { confidence: 0 });
  }
});

document.querySelectorAll(".tip-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    chatInput.value = chip.dataset.sample;
    chatInput.focus();
  });
});

// CONTROLES SUPERIORES
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
  try {
    const res = await fetch("/api/self_teach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.ok && data.brain_state) updateBrainUI(data.brain_state);
  } catch (e) {}
}

btnManualSelfPlay.addEventListener("click", triggerSelfPlayStep);

setInterval(() => {
  if (autoSelfPlayToggle && autoSelfPlayToggle.checked) triggerSelfPlayStep();
}, 9000);

btnTrainEpochs.addEventListener("click", async () => {
  btnTrainEpochs.disabled = true;
  const orig = btnTrainEpochs.textContent;
  btnTrainEpochs.textContent = "⏳...";
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
        `⚡ **+20 épocas entrenadas en el Micro-Transformer (Q, K, V + MLP).** Loss actual: \`${data.final_loss.toFixed(5)}\`.`,
        { mode: "neural_hybrid", confidence: 0.99, loss: data.final_loss }
      );
    }
  } finally {
    btnTrainEpochs.disabled = false;
    btnTrainEpochs.textContent = orig;
  }
});

btnResetMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  resetMenu.classList.toggle("open");
});
document.addEventListener("click", () => resetMenu.classList.remove("open"));

btnResetSeed.addEventListener("click", async () => {
  if (!confirm("¿Reiniciar cerebro y biblioteca RAG con la semilla inicial?")) return;
  const res = await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_seed: true }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
  chatMessages.innerHTML = "";
  appendMessage("ai", "🌱 " + data.message, { mode: "neural_hybrid", confidence: 1.0 });
});

btnResetBlank.addEventListener("click", async () => {
  if (!confirm("¿Dejar todo 100% en blanco (Tabula Rasa)?")) return;
  const res = await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_seed: false }),
  });
  const data = await res.json();
  if (data.brain_state) updateBrainUI(data.brain_state);
  chatMessages.innerHTML = "";
  appendMessage("ai", "🧪 **Tabula Rasa activada.** Cerebro y documentos en cero absoluto.", {
    mode: "auto_self_taught",
    confidence: 1.0,
  });
});

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

directTeachForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const stim = document.getElementById("teachStimulus").value.trim();
  const resp = document.getElementById("teachResponse").value.trim();
  if (!stim || !resp) return;
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
});

// RENDERIZADO GENERAL DE TELEMETRÍA
function updateBrainUI(state) {
  const m = state.metrics;

  metricParams.textContent = m.total_parameters.toLocaleString();
  metricVocab.textContent = m.vocab_size.toLocaleString();
  metricRagTasks.textContent = `${m.rag_docs_count || 0} / ${m.tasks_pending || 0}`;
  metricSteps.textContent = m.steps_trained.toLocaleString();
  metricLoss.textContent = Number(m.last_loss || 0).toFixed(4);
  epochsLabel.textContent = `${m.epochs_trained} épocas`;

  if (state.architecture) {
    const a = state.architecture;
    archLabel.textContent = `${a.input_dim} → ${a.hidden1_dim} → ${a.hidden2_dim} → ${a.output_dim}`;
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

  renderTokenAttention(state.token_attention || []);
  renderRagDocs(state.rag_docs || []);
  renderAgentTasksAndNotes(state.tasks || [], state.notes || []);
  renderFactsAndRules(state.facts || [], state.logic_rules || []);
  renderSelfTeachLog(state.self_teaching_log || []);
  renderEpisodesList(state.episodes || []);
  renderVocabCloud(state.top_vocab || []);
  drawNeuralNetwork(state.activations);
  drawLossCurve(state.loss_history || []);
}

function renderTokenAttention(attnList) {
  if (!attnList || attnList.length === 0) return;
  attnTokens.innerHTML = "";
  const maxW = Math.max(...attnList.map((a) => a.weight), 0.01);
  attnList.forEach((item) => {
    const intensity = Math.min(1, item.weight / maxW);
    const chip = document.createElement("span");
    chip.className = "attn-chip";
    chip.style.backgroundColor = `rgba(59, 130, 246, ${0.15 + intensity * 0.7})`;
    chip.title = `Peso Self-Attention: ${(item.weight * 100).toFixed(1)}%`;
    chip.textContent = `${item.token} (${Math.round(item.weight * 100)}%)`;
    attnTokens.appendChild(chip);
  });
}

function renderRagDocs(docs) {
  ragDocsList.innerHTML = "";
  if (docs.length === 0) {
    ragDocsList.innerHTML = `<div class="tab-help">Aún no hay documentos estudiados. Sube un archivo con <strong>📎 Subir Doc</strong> o investiga un tema arriba.</div>`;
    return;
  }
  docs.forEach((d) => {
    const icon = d.source_type === "wikipedia" ? "🌐" : "📄";
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">${icon} ${escapeHtml(d.title)}</span>
        <span class="resp-tag">${d.chunks_count} fragmentos vectoriales · ${escapeHtml(d.date)}</span>
      </div>
      <div class="item-badges">
        <button class="btn-del" title="Eliminar documento" data-id="${d.id}">✕</button>
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

function renderAgentTasksAndNotes(tasks, notes) {
  agentTasksNotesList.innerHTML = "";
  if (tasks.length === 0 && notes.length === 0) {
    agentTasksNotesList.innerHTML = `<div class="tab-help">Sin tareas ni notas. Pídele en el chat: «Recuérdame entregar proyecto» o «Guarda nota: ...».</div>`;
    return;
  }

  tasks.forEach((t) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <label class="list-item-main" style="flex-direction: row; align-items: center; gap: 8px; cursor: pointer;">
        <input type="checkbox" ${t.done ? "checked" : ""} />
        <span class="${t.done ? "task-done-text" : "resp-tag"}">[#${t.id}] ${escapeHtml(t.description)}</span>
      </label>
      <button class="btn-del" title="Borrar tarea">✕</button>
    `;
    item.querySelector("input").addEventListener("change", async (e) => {
      const res = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "toggle", id: t.id, done: e.target.checked }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    item.querySelector(".btn-del").addEventListener("click", async () => {
      const res = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "delete", id: t.id }),
      });
      const data = await res.json();
      if (data.brain_state) updateBrainUI(data.brain_state);
    });
    agentTasksNotesList.appendChild(item);
  });

  notes.forEach((n) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">📝 Nota #${n.id}</span>
        <span class="resp-tag">${escapeHtml(n.content)}</span>
      </div>
      <button class="btn-del" title="Borrar nota">✕</button>
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
    agentTasksNotesList.appendChild(item);
  });
}

function renderFactsAndRules(facts, rules) {
  factsList.innerHTML = "";
  if (facts.length === 0 && rules.length === 0) {
    factsList.innerHTML = `<div class="tab-help">Sin hechos ni reglas lógicas aún.</div>`;
    return;
  }

  rules.forEach((r) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">🧩 Regla Condicional #${r.id}</span>
        <span class="resp-tag">Si <strong>${escapeHtml(r.condition)}</strong> ➔ entonces <strong>${escapeHtml(r.consequence)}</strong></span>
      </div>
    `;
    factsList.appendChild(item);
  });

  facts.forEach((f) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">📌 ${escapeHtml(f.subject)} → <em>${escapeHtml(f.relation)}</em></span>
        <span class="resp-tag">${escapeHtml(f.object)}</span>
      </div>
      <button class="btn-del" data-id="${f.id}">✕</button>
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
      typeof entry.loss === "number" ? `<span class="reward-pill">Loss: ${entry.loss.toFixed(4)}</span>` : "";
    item.innerHTML = `
      <div class="list-item-main">
        <span class="stim-tag">[${escapeHtml(entry.time)}] ${escapeHtml(entry.summary)}</span>
        <span class="resp-tag">${escapeHtml(entry.detail)}</span>
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
        <span class="stim-tag">🗣️ "${escapeHtml(ep.stimulus)}"</span>
        <span class="resp-tag">🤖 "${escapeHtml(ep.response)}"</span>
      </div>
      <div class="item-badges">
        <span class="reward-pill">⚖️ ${Number(ep.reward).toFixed(2)}</span>
        <button class="btn-del" data-id="${ep.id}">✕</button>
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
  vocab.slice(0, 28).forEach((v) => {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    chip.textContent = `${v.word} (${v.freq})`;
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
          .map((n) => `<strong>${escapeHtml(n.word)}</strong> (${n.similarity})`)
          .join(", ");
        neighborsResult.innerHTML = `🧭 Vecinos de <strong>«${escapeHtml(v.word)}»</strong> en <code>W_emb</code>: ${list}`;
      } else {
        neighborsResult.innerHTML = `🧭 Pocos vecinos aún para <strong>«${escapeHtml(v.word)}»</strong>.`;
      }
    });
    vocabCloud.appendChild(chip);
  });
}

function drawNeuralNetwork(activations) {
  const w = nnCanvas.width;
  const h = nnCanvas.height;
  nnCtx.clearRect(0, 0, w, h);

  const layers = [
    { name: "Attn Q·K·V", vals: (activations && activations.input) || new Array(8).fill(0.1), count: 8 },
    { name: "Swish H1", vals: (activations && activations.hidden1) || new Array(10).fill(0.1), count: 10 },
    { name: "Res+Tanh", vals: (activations && activations.hidden2) || new Array(8).fill(0.1), count: 8 },
    { name: "Latent", vals: (activations && activations.output) || new Array(8).fill(0.1), count: 8 },
  ];

  const colSpacing = w / (layers.length + 0.4);
  const positions = [];

  layers.forEach((layer, lIdx) => {
    const x = colSpacing * (lIdx + 0.7);
    const nodes = [];
    const n = layer.count;
    const rowSpacing = (h - 24) / (n + 1);
    for (let i = 0; i < n; i++) {
      const y = 8 + rowSpacing * (i + 1);
      const val = layer.vals[i % layer.vals.length] || 0;
      nodes.push({ x, y, val });
    }
    positions.push({ name: layer.name, x, nodes });
  });

  for (let l = 0; l < positions.length - 1; l++) {
    const curr = positions[l].nodes;
    const next = positions[l + 1].nodes;
    for (let i = 0; i < curr.length; i++) {
      for (let j = 0; j < next.length; j++) {
        const strength = Math.min(1, Math.abs(curr[i].val * next[j].val) * 2.5 + 0.08);
        const isPos = curr[i].val * next[j].val >= 0;
        nnCtx.strokeStyle = isPos
          ? `rgba(59, 130, 246, ${strength * 0.55})`
          : `rgba(139, 92, 246, ${strength * 0.45})`;
        nnCtx.lineWidth = strength * 1.4;
        nnCtx.beginPath();
        nnCtx.moveTo(curr[i].x, curr[i].y);
        nnCtx.lineTo(next[j].x, next[j].y);
        nnCtx.stroke();
      }
    }
  }

  positions.forEach((layer) => {
    layer.nodes.forEach((node) => {
      const mag = Math.min(1, Math.abs(node.val));
      const radius = 3.8 + mag * 2.4;
      nnCtx.beginPath();
      nnCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      nnCtx.fillStyle =
        node.val >= 0
          ? `rgba(16, 185, 129, ${0.35 + mag * 0.65})`
          : `rgba(245, 158, 11, ${0.35 + mag * 0.65})`;
      nnCtx.fill();
      nnCtx.strokeStyle = "#e2e8f0";
      nnCtx.lineWidth = 0.7;
      nnCtx.stroke();
    });
    nnCtx.fillStyle = "#94a3b8";
    nnCtx.font = "10px Segoe UI, sans-serif";
    nnCtx.textAlign = "center";
    nnCtx.fillText(layer.name, layer.x, h - 3);
  });
}

function drawLossCurve(lossHistory) {
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  lossCtx.clearRect(0, 0, w, h);

  lossCtx.strokeStyle = "#1e293b";
  lossCtx.lineWidth = 1;
  for (let y = 20; y < h - 18; y += 32) {
    lossCtx.beginPath();
    lossCtx.moveTo(30, y);
    lossCtx.lineTo(w - 10, y);
    lossCtx.stroke();
  }

  if (!lossHistory || lossHistory.length === 0) return;

  const maxLoss = Math.max(...lossHistory, 0.25);
  const padLeft = 34, padRight = 12, padTop = 14, padBottom = 18;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  lossCtx.beginPath();
  lossHistory.forEach((val, idx) => {
    const x = padLeft + (lossHistory.length === 1 ? plotW / 2 : (idx / (lossHistory.length - 1)) * plotW);
    const normY = val / (maxLoss + 1e-6);
    const y = padTop + plotH * (1 - Math.min(1, Math.max(0, normY)));
    if (idx === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  });
  lossCtx.strokeStyle = "#10b981";
  lossCtx.lineWidth = 2.1;
  lossCtx.stroke();

  const lastVal = lossHistory[lossHistory.length - 1];
  lossCtx.fillStyle = "#34d399";
  lossCtx.font = "10px Segoe UI, sans-serif";
  lossCtx.textAlign = "right";
  lossCtx.fillText(`Loss: ${lastVal.toFixed(4)}`, w - 10, padTop + 8);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function initApp() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    if (data.ok && data.brain_state) {
      updateBrainUI(data.brain_state);
      appendMessage(
        "ai",
        "🚀 **¡Bienvenido a NeuroZero 2.0 (Paquete Completo)!**\n" +
          "• **Cerebro Micro-Transformer:** Con *Self-Attention* (`Q·Kᵀ·V`), raíces morfológicas y Backpropagation en NumPy.\n" +
          "• **Super-Aprendizaje RAG y Web:** Súbeme archivos con **📎 Subir Doc** o dime *«Investiga sobre [cualquier tema]»* para estudiarlo sola en Wikipedia.\n" +
          "• **Inferencia Lógica:** Encadeno silogismos (*A es B* + *B es C* ➔ *A es C*) y reglas condicionales (*Si X entonces Y*).\n" +
          "• **Agente Útil:** Resuelvo cálculos/ecuaciones, convierto unidades, gestiono tus tareas/notas y creo archivos.",
        { mode: "neural_hybrid", confidence: 1.0 }
      );
    }
  } catch (err) {
    console.error("Error inicializando NeuroZero 2.0:", err);
  }
}

initApp();
