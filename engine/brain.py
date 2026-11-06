"""
Orquestador Cognitivo NeuroZero 2.0 (Paquete Completo desde Cero):
Integra los 4 Pilares en un solo cerebro:
1. Micro-Transformer en NumPy con Self-Attention (Q, K, V) + Morfología Español + Backprop/Adam.
2. Super-Aprendizaje Masivo: Base Vectorial RAG en SQLite + Lector de Documentos (.txt, .md, .docx,
   .py, .csv, .pdf) + Auto-Investigador Autónomo en Wikipedia.
3. Agente Útil con Herramientas: Calculadora AST, Álgebra, Conversiones, Tareas/Recordatorios,
   Notas Personales y Generador de Archivos.
4. Motor de Razonamiento e Inferencia Lógica (Silogismos multi-salto y Modus Ponens).
"""

import os
import re
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

from .tokenizer import ScratchTokenizer, NaturalChatParser, strip_accents, spanish_stem, STOPWORDS_ES
from .neural_net import ScratchNeuralNetwork
from .memory import HybridMemorySystem, EpisodicMemoryItem
from .reasoning import LogicalReasoningEngine
from .knowledge_rag import ScratchVectorRAG, DocumentParserFromScratch, WikipediaAutoResearcher
from .tools_agent import ScratchAgentToolkit


INITIAL_SEED_KNOWLEDGE = [
    ("hola", "¡Hola! Soy NeuroZero 2.0. Ahora cuento con Micro-Transformer (Self-Attention), Motor Lógico, Base Documental RAG, Auto-Investigación en Wikipedia y Herramientas de Agente. ¿Qué hacemos o aprendemos hoy?"),
    ("buenos dias", "¡Buenos días! Mis capas de atención y herramientas están listas. ¿Qué proyecto, cálculo, tarea o tema vemos hoy?"),
    ("buenas tardes", "¡Buenas tardes! Puedes platicar conmigo, subirme archivos para que los estudie, pedirme cálculos o gestionar tus tareas."),
    ("buenas noches", "¡Buenas noches! Seguimos entrenando o resolviendo lo que necesites."),
    ("como estas", "¡Excelente! Con mi nueva arquitectura Self-Attention y herramientas activas. ¿Y tú cómo estás?"),
    ("que puedes hacer", "Puedo: 1) Auto-enseñarme de lo que hablamos, 2) Leer documentos (.txt, .md, .docx, .py, .csv, .pdf) e investigar sola en Wikipedia, 3) Razonar lógicamente encadenando hechos, y 4) Calcular matemáticas/ecuaciones, gestionar tus tareas/notas y crear archivos."),
    ("quien eres", "Soy NeuroZero 2.0, un motor de IA híbrido neuro-simbólico con Self-Attention, RAG y herramientas construido desde cero en Python y NumPy."),
    ("gracias", "¡Con gusto! Cada consulta y documento fortalece mis pesos neuronales."),
    ("adios", "¡Hasta pronto! Todo tu conocimiento, documentos, tareas y pesos quedan guardados en disco."),
]


class ScratchAIEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.exports_dir = os.path.join(os.path.dirname(self.data_dir), "exports")
        os.makedirs(self.exports_dir, exist_ok=True)

        self.weights_path = os.path.join(self.data_dir, "brain_weights_v2.npz")
        self.memory_path = os.path.join(self.data_dir, "brain_memory_v2.json")
        self.rag_db_path = os.path.join(self.data_dir, "rag_knowledge.db")
        self.agent_db_path = os.path.join(self.data_dir, "assistant_data.db")

        self.tokenizer = ScratchTokenizer()
        self.memory = HybridMemorySystem()
        self.reasoning = LogicalReasoningEngine()
        self.rag = ScratchVectorRAG(db_path=self.rag_db_path)
        self.toolkit = ScratchAgentToolkit(db_path=self.agent_db_path, exports_dir=self.exports_dir)

        self.nn = ScratchNeuralNetwork(
            vocab_size=len(self.tokenizer.word2idx),
            embed_dim=64,
            hidden1_dim=128,
            hidden2_dim=96,
            learning_rate=0.006,
        )

        # Configuraciones activas
        self.curiosity_mode: bool = False
        self.auto_self_play: bool = True
        self.auto_web_research: bool = True  # Investiga sola en Wikipedia cuando le preguntas algo que desconoce
        self.confidence_threshold: float = 0.44

        # Contexto conversacional
        self.pending_unknown_stimulus: Optional[str] = None
        self.pending_unknown_context: str = ""
        self.last_user_stimulus: str = ""
        self.last_ai_response: str = ""
        self.last_ai_question_concept: Optional[str] = None
        self.last_episode_id: Optional[int] = None
        self.conversation_context: str = ""

        if os.path.exists(self.memory_path) and os.path.exists(self.weights_path):
            self.load_state()
        else:
            self.seed_initial_brain()

    def seed_initial_brain(self):
        self.memory.add_or_update_fact("ia", "nombre", "NeuroZero 2.0", category="ai")
        self.memory.add_or_update_fact(
            "ia",
            "identidad",
            "un motor híbrido con Micro-Transformer Self-Attention, RAG, Inferencia Lógica y Agente de Herramientas hecho desde cero",
            category="ai",
        )
        for stim, resp in INITIAL_SEED_KNOWLEDGE:
            self._register_and_encode(stim)
            self._register_and_encode(resp)
            self.memory.add_episode(stimulus=stim, response=resp, context="", reward=1.25, source="seed")

        self.memory.log_self_teaching(
            kind="init",
            summary="NeuroZero 2.0 Inicializado",
            detail="Micro-Transformer (Q,K,V) + Motor Lógico + RAG + Herramientas listos.",
        )
        self.train_epochs(epochs=8)
        self.save_state()

    def reset_brain(self, include_seed: bool = True):
        self.tokenizer = ScratchTokenizer()
        self.memory = HybridMemorySystem()
        self.reasoning = LogicalReasoningEngine()
        self.rag.clear_all()
        self.nn = ScratchNeuralNetwork(
            vocab_size=len(self.tokenizer.word2idx),
            embed_dim=64,
            hidden1_dim=128,
            hidden2_dim=96,
            learning_rate=0.006,
        )
        self.pending_unknown_stimulus = None
        self.pending_unknown_context = ""
        self.last_user_stimulus = ""
        self.last_ai_response = ""
        self.last_ai_question_concept = None
        self.last_episode_id = None
        self.conversation_context = ""

        if include_seed:
            self.seed_initial_brain()
        else:
            self.memory.log_self_teaching(
                kind="reset",
                summary="Tabula Rasa Completa",
                detail="Cerebro y biblioteca documental reiniciados desde cero absoluto.",
            )
            self.save_state()

    def _register_and_encode(self, text: str, record_attention: bool = False) -> List[int]:
        tokens_str = self.tokenizer.tokenize(text, learn_vocab=True)
        ids = [self.tokenizer.word2idx.get(t, 1) for t in tokens_str]
        self.nn.expand_vocab(len(self.tokenizer.word2idx))
        self.nn.update_cooccurrence_embeddings(ids)
        if record_attention and ids:
            self.nn.encode_tokens(ids, tokens_str=tokens_str, record_attention=True)
        return ids

    # =========================================================================
    # PILAR 1: INGESTA MASIVA DE DOCUMENTOS Y AUTO-INVESTIGACIÓN WIKIPEDIA
    # =========================================================================
    def ingest_document_bytes(self, filename: str, raw_bytes: bytes) -> Dict[str, object]:
        text = DocumentParserFromScratch.parse_bytes(filename, raw_bytes)
        if not text or len(text.strip()) < 15:
            return {
                "ok": False,
                "message": f"No se pudo extraer texto legible de `{filename}` (verifica que no sea un PDF escaneado como imagen).",
            }
        return self.ingest_text_document(title=filename, text=text, source_type="file")

    def ingest_text_document(self, title: str, text: str, source_type: str = "file") -> Dict[str, object]:
        """
        Lee un texto completo (archivo o artículo de Wikipedia):
        1. Lo divide en fragmentos y lo indexa en la base vectorial RAG (SQLite).
        2. Registra el vocabulario nuevo y entrena los vectores de co-ocurrencia en NumPy.
        3. Extrae hechos semánticos estructurados de las oraciones hacia el Grafo de Conocimiento.
        4. Entrena pasos de Backpropagation sobre las definiciones principales del texto.
        """
        res = self.rag.ingest_document(title=title, text=text, source_type=source_type)
        if not res.get("ok"):
            return {"ok": False, "message": "El texto es demasiado corto para indexarse."}

        chunks: List[str] = res.get("chunks", [])
        facts_extracted = 0
        trained_pairs = 0
        last_loss = self.nn.loss_history[-1] if self.nn.loss_history else 0.0

        # Aprender vocabulario y extraer hechos de los primeros 25 fragmentos
        for chunk in chunks[:25]:
            self._register_and_encode(chunk, record_attention=False)
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for sent in sentences[:4]:
                sent_clean = sent.strip()
                if 15 <= len(sent_clean) <= 220:
                    fact = NaturalChatParser.extract_fact(sent_clean)
                    if fact and fact["category"] == "world":
                        self.memory.add_or_update_fact(
                            fact["subject"], fact["relation"], fact["object"], category="world"
                        )
                        facts_extracted += 1

        # Auto-entrenar la red neuronal sobre el resumen inicial del documento
        clean_title = os.path.splitext(title)[0].replace("_", " ").strip()
        if chunks:
            first_summary = chunks[0][:320]
            _, last_loss = self.learn_association(
                stimulus=f"que dice {clean_title}",
                response=f"Según «{title}»: {first_summary}",
                reward=1.4,
                source="document_rag",
                steps=5,
            )
            trained_pairs += 1

        self.memory.log_self_teaching(
            kind="rag_ingest",
            summary=f"Estudió «{title}» ({source_type})",
            detail=f"{len(chunks)} fragmentos RAG indexados, +{facts_extracted} hechos al grafo.",
            loss=last_loss,
        )
        self.save_state()

        return {
            "ok": True,
            "title": title,
            "source_type": source_type,
            "chunks_added": len(chunks),
            "facts_extracted": facts_extracted,
            "summary": chunks[0][:360] if chunks else "",
            "loss": round(last_loss, 5),
            "brain_state": self.get_brain_telemetry(),
        }

    def research_topic_online(self, query: str) -> Dict[str, object]:
        """Investiga un tema en Wikipedia, lo guarda en RAG + Grafo, entrena neuronas y resume."""
        wiki_data = WikipediaAutoResearcher.search_and_fetch(query)
        if not wiki_data:
            return {
                "ok": False,
                "message": "No pude encontrar un artículo en Wikipedia para ese término (o no hay conexión a Internet).",
            }

        ingested = self.ingest_text_document(
            title=f"Wiki: {wiki_data['title']}",
            text=wiki_data["content"],
            source_type="wikipedia",
        )
        # También entrenar pregunta directa sobre el título
        summary_text = wiki_data["content"].split("\n")[0][:420]
        ep, loss = self.learn_association(
            stimulus=f"que es {wiki_data['title'].lower()}",
            response=summary_text,
            reward=1.5,
            source="wikipedia_research",
            steps=6,
        )
        self.save_state()
        return {
            "ok": True,
            "title": wiki_data["title"],
            "url": wiki_data["url"],
            "summary": summary_text,
            "chunks_added": ingested.get("chunks_added", 0),
            "facts_extracted": ingested.get("facts_extracted", 0),
            "episode_id": ep.id,
            "loss": round(loss, 5),
            "brain_state": self.get_brain_telemetry(),
        }

    # =========================================================================
    # EVALUACIÓN DEL MICRO-TRANSFORMER Y APRENDIZAJE CONTINUO
    # =========================================================================
    def _lexical_overlap_score(self, query_text: str, candidate_text: str) -> float:
        q_tokens = self.tokenizer.tokenize(query_text, learn_vocab=False)
        c_tokens = self.tokenizer.tokenize(candidate_text, learn_vocab=False)
        if not q_tokens or not c_tokens:
            return 0.0
        if q_tokens == c_tokens:
            return 1.0

        total_weight = 0.0
        matched_weight = 0.0
        for qt in q_tokens:
            w = 0.35 if qt in STOPWORDS_ES else 1.2
            total_weight += w
            if qt in c_tokens:
                matched_weight += w
            else:
                best_sim = max((self.tokenizer.word_similarity(qt, ct) for ct in c_tokens), default=0.0)
                if best_sim >= 0.62:
                    matched_weight += w * best_sim

        forward_score = matched_weight / max(total_weight, 1e-6)
        len_ratio = min(len(q_tokens), len(c_tokens)) / max(len(q_tokens), len(c_tokens))
        return float(forward_score * (0.7 + 0.3 * len_ratio))

    def evaluate_candidates(
        self, user_text: str, context_text: str = ""
    ) -> Tuple[Optional[EpisodicMemoryItem], float, Dict[str, float], List[Dict[str, object]]]:
        tokens_str = self.tokenizer.tokenize(user_text, learn_vocab=False)
        msg_ids = self.tokenizer.encode(user_text, learn_vocab=False)
        ctx_ids = self.tokenizer.encode(context_text, learn_vocab=False) if context_text else []

        msg_vec = self.nn.encode_tokens(msg_ids, tokens_str=tokens_str, record_attention=True)
        ctx_vec = self.nn.encode_tokens(ctx_ids) if ctx_ids else np.zeros_like(msg_vec)

        pred_resp_vec, _ = self.nn.forward(msg_vec, ctx_vec, record_telemetry=True)

        if not self.memory.episodes:
            return None, 0.0, {"neural": 0.0, "embedding": 0.0, "lexical": 0.0}, []

        scored_candidates = []
        for ep in self.memory.episodes:
            if ep.reward <= 0.15:
                continue

            ep_stim_ids = self.tokenizer.encode(ep.stimulus, learn_vocab=False)
            ep_resp_ids = self.tokenizer.encode(ep.response, learn_vocab=False)

            ep_stim_vec = self.nn.encode_tokens(ep_stim_ids)
            ep_resp_vec = self.nn.encode_tokens(ep_resp_ids)

            neural_sim = float(np.clip(np.sum(pred_resp_vec * ep_resp_vec), 0.0, 1.0))
            emb_sim = float(np.clip(np.sum(msg_vec * ep_stim_vec), 0.0, 1.0))
            lex_sim = self._lexical_overlap_score(user_text, ep.stimulus)

            ctx_bonus = 0.0
            if context_text and ep.context:
                ctx_bonus = 0.08 * self._lexical_overlap_score(context_text, ep.context)

            reward_factor = float(np.clip(0.75 + 0.25 * ep.reward, 0.4, 1.35))
            raw_confidence = (0.38 * neural_sim) + (0.24 * emb_sim) + (0.38 * lex_sim) + ctx_bonus
            if lex_sim >= 0.88:
                raw_confidence = max(raw_confidence, 0.84 + 0.10 * neural_sim)

            final_score = float(np.clip(raw_confidence * reward_factor, 0.0, 0.99))
            scored_candidates.append(
                {
                    "episode": ep,
                    "score": final_score,
                    "breakdown": {
                        "neural": round(neural_sim, 3),
                        "embedding": round(emb_sim, 3),
                        "lexical": round(lex_sim, 3),
                        "reward": round(ep.reward, 2),
                    },
                }
            )

        if not scored_candidates:
            return None, 0.0, {"neural": 0.0, "embedding": 0.0, "lexical": 0.0}, []

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        best = scored_candidates[0]
        top_list = [
            {
                "id": c["episode"].id,
                "stimulus": c["episode"].stimulus,
                "response": c["episode"].response,
                "score": round(c["score"] * 100, 1),
                "breakdown": c["breakdown"],
            }
            for c in scored_candidates[:3]
        ]
        return best["episode"], best["score"], best["breakdown"], top_list

    def learn_association(
        self,
        stimulus: str,
        response: str,
        context: str = "",
        reward: float = 1.25,
        source: str = "auto_self_taught",
        steps: int = 5,
    ) -> Tuple[EpisodicMemoryItem, float]:
        stim_ids = self._register_and_encode(stimulus)
        resp_ids = self._register_and_encode(response)
        ctx_ids = self._register_and_encode(context) if context else []

        episode = self.memory.add_episode(
            stimulus=stimulus,
            response=response,
            context=context,
            reward=reward,
            source=source,
        )

        last_loss = 0.0
        other_episodes = [ep for ep in self.memory.episodes if ep.id != episode.id]
        for _ in range(steps):
            neg_ids = None
            if other_episodes:
                neg_ep = random.choice(other_episodes)
                neg_ids = self.tokenizer.encode(neg_ep.response, learn_vocab=False)
            last_loss = self.nn.train_step(
                msg_token_ids=stim_ids,
                target_token_ids=resp_ids,
                ctx_token_ids=ctx_ids,
                negative_token_ids=neg_ids,
                reward_weight=max(0.5, reward),
            )

        return episode, last_loss

    def auto_learn_from_user_message(
        self, user_message: str, keywords: List[str], extracted_fact: Optional[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], float]:
        learned_events: List[Dict[str, str]] = []
        last_loss = self.nn.loss_history[-1] if self.nn.loss_history else 0.0

        self.memory.absorb_user_statement_into_concepts(user_message, keywords)

        if not NaturalChatParser.is_question(user_message) and len(user_message.split()) >= 2:
            if self.last_ai_question_concept:
                concept = self.last_ai_question_concept
                shifted_resp = NaturalChatParser.shift_user_to_second_person(user_message)
                _, last_loss = self.learn_association(
                    stimulus=f"que sabes de {concept}",
                    response=f"Sobre {concept}, me platicaste que {shifted_resp}.",
                    context=self.last_ai_response,
                    reward=1.35,
                    source="conversational_mirror",
                    steps=4,
                )
                desc = f"Espejo conversacional sobre «{concept}» → «{shifted_resp}»"
                self.memory.log_self_teaching("mirror", f"Aprendió respuesta sobre «{concept}»", desc, last_loss)
                learned_events.append({"type": "mirror", "text": desc})
                self.last_ai_question_concept = None

        synthetic_pairs = NaturalChatParser.generate_self_teaching_pairs(
            user_message, keywords, extracted_fact=extracted_fact
        )
        for trig, resp, desc in synthetic_pairs:
            _, last_loss = self.learn_association(
                stimulus=trig,
                response=resp,
                context=self.conversation_context,
                reward=1.3,
                source="auto_self_taught",
                steps=4,
            )
            self.memory.log_self_teaching("self_taught", f"Auto-pregunta: «{trig}»", f"→ «{resp}»", last_loss)
            learned_events.append({"type": "self_taught", "text": f"«{trig}» → «{resp}»"})

        return learned_events, last_loss

    def synthesize_autonomous_reply(
        self,
        user_message: str,
        keywords: List[str],
        saved_fact: Optional[Dict[str, object]],
        best_ep: Optional[EpisodicMemoryItem],
        confidence: float,
    ) -> str:
        is_q = NaturalChatParser.is_question(user_message)
        shifted_msg = NaturalChatParser.shift_user_to_second_person(user_message)

        if saved_fact is not None:
            cat = saved_fact["category"]
            rel = str(saved_fact["relation"])
            obj = str(saved_fact["object"])
            subj = str(saved_fact["subject"])

            if cat == "user" and rel == "nombre":
                return f"¡Mucho gusto, {obj}! Ya registré tu nombre en mi Grafo Semántico."
            if cat == "user" and rel == "gusto":
                self.last_ai_question_concept = keywords[-1] if keywords else obj
                return f"¡Genial que te guste {obj}! Ya lo entrené en mi memoria. ¿Qué es lo mejor de {obj} para ti?"
            if cat == "user" and rel == "estado_actual":
                self.last_ai_question_concept = keywords[-1] if keywords else obj
                return f"¡Excelente! Registré que estás {obj}. ¿Cómo vas con eso?"
            if cat == "ai" and rel == "nombre":
                return f"¡Me encanta! A partir de ahora me llamo {obj}."
            if cat == "world":
                # Verificar si este nuevo hecho dispara deducciones lógicas nuevas!
                deductions = self.reasoning.deduce_all_about_entity(subj, self.memory.facts)
                ded_note = ""
                if deductions:
                    ded_note = f"\n🧩 *¡Además acabo de deducir por lógica que: **{deductions[0]['conclusion']}**!*"
                self.last_ai_question_concept = subj.lower()
                return (
                    f"¡Aprendido! Guardé en mi grafo que «{subj} {rel} {obj}».{ded_note}\n"
                    f"¿Qué más sabes o quieres explorar sobre {subj.lower()}?"
                )

        if is_q:
            for kw in keywords:
                related = self.memory.get_related_concepts(kw, top_k=3)
                markov_synth = self.memory.generate_sentence_from_seed([kw])
                if markov_synth:
                    return f"Por lo que he aprendido sobre «{kw}»: {markov_synth}."
                if related:
                    self.last_ai_question_concept = kw
                    return f"Tengo «{kw}» vinculado con {', '.join(related)}, pero aún puedo aprender más. Cuéntame o dime *«Investiga sobre {kw}»*."

            if best_ep is not None and confidence >= 0.25:
                return best_ep.response

            main_kw = keywords[0] if keywords else user_message
            self.last_ai_question_concept = main_kw
            return f"Aún no tengo datos sobre «{main_kw}». Platícame al respecto, súbeme un documento con 📎 o dime *«Investiga sobre {main_kw}»*."

        main_kw = keywords[0] if keywords else ""
        if main_kw:
            self.last_ai_question_concept = main_kw
            related = self.memory.get_related_concepts(main_kw, top_k=2)
            rel_note = f" (vinculado en mis neuronas con {', '.join(related)})" if related else ""
            return f"¡Interesante! Ya absorbí en mi red que {shifted_msg}{rel_note}. ¿Qué más me cuentas sobre {main_kw}?"

        return f"He procesado «{user_message}» en mis capas de atención. ¿En qué más trabajamos?"

    # =========================================================================
    # CICLO COGNITIVO COMPLETO NEUROZERO 2.0
    # =========================================================================
    def process_chat_message(self, user_message: str) -> Dict[str, object]:
        user_message = user_message.strip()
        if not user_message:
            return {"reply": "Escribe un mensaje, cálculo, tarea o sube un documento.", "confidence": 0.0}

        # Activar Self-Attention sobre los tokens del mensaje para telemetría visual
        self._register_and_encode(user_message, record_attention=True)
        keywords = self.tokenizer.extract_keywords(user_message)
        norm_msg = strip_accents(user_message.lower())

        # 1. ¿Modo de pregunta manual pendiente?
        if self.pending_unknown_stimulus is not None:
            if norm_msg in {"no se", "olvidalo", "nada", "pasar", "saltar", "cancelar"}:
                skipped = self.pending_unknown_stimulus
                self.pending_unknown_stimulus = None
                return {
                    "reply": f"Saltamos «{skipped}». ¡Sigamos!",
                    "confidence": 1.0,
                    "mode": "curiosity_cancelled",
                    "brain_state": self.get_brain_telemetry(),
                }
            learned_stim = self.pending_unknown_stimulus
            self.pending_unknown_stimulus = None
            ep, loss = self.learn_association(learned_stim, user_message, reward=1.45, source="curiosity", steps=8)
            self.save_state()
            return {
                "reply": f"🧠 ¡Aprendido! Cuando digas «{learned_stim}», responderé «{user_message}» (Loss: {loss:.4f}).",
                "confidence": 0.99,
                "mode": "learned_from_curiosity",
                "episode_id": ep.id,
                "loss": round(loss, 5),
                "brain_state": self.get_brain_telemetry(),
            }

        # 2. ¿Enseñanza explícita ("Cuando te diga X responde Y") o Corrección ("No, responde...")?
        explicit_pair = NaturalChatParser.detect_explicit_teaching(user_message)
        if explicit_pair is not None:
            trig, resp = explicit_pair
            ep, loss = self.learn_association(trig, resp, reward=1.5, source="chat_taught", steps=8)
            self.save_state()
            return {
                "reply": f"⚡ ¡Regla entrenada en el Micro-Transformer! «{trig}» → «{resp}» (Loss: {loss:.4f}).",
                "confidence": 0.99,
                "mode": "explicit_teaching",
                "episode_id": ep.id,
                "loss": round(loss, 5),
                "brain_state": self.get_brain_telemetry(),
            }

        correction = NaturalChatParser.detect_chat_correction(user_message)
        if correction is not None and self.last_user_stimulus:
            res = self.apply_reinforcement(self.last_episode_id, "correct", corrected_response=correction)
            return {
                "reply": f"🔧 ¡Corrección aplicada! {res.get('message', '')}",
                "confidence": 0.99,
                "mode": "chat_correction",
                "episode_id": self.last_episode_id,
                "brain_state": self.get_brain_telemetry(),
            }

        # 3. PILAR AGENTE ÚTIL: ¿Es un cálculo matemático, ecuación, conversión, tarea, nota o archivo?
        tool_result = self.toolkit.try_execute_tool(user_message)
        if tool_result is not None:
            msg_vec = self.nn.encode_tokens(self.tokenizer.encode(user_message))
            self.nn.forward(msg_vec, record_telemetry=True)
            self.last_user_stimulus = user_message
            self.last_ai_response = str(tool_result["reply"])
            return {
                "reply": tool_result["reply"],
                "confidence": 1.0,
                "mode": "agent_tool",
                "tool_used": tool_result["tool_name"],
                "brain_state": self.get_brain_telemetry(),
            }

        # 4. PILAR SUPER-APRENDIZAJE: ¿Orden directa de investigar en Wikipedia?
        if re.match(r"^(?:investiga(?:\s+sobre)?|busca\s+en\s+wikipedia(?:\s+sobre)?|estudia\s+sobre)\s+(.+)$", norm_msg):
            res_wiki = self.research_topic_online(user_message)
            if res_wiki.get("ok"):
                reply = (
                    f"🌐 **Auto-Investigación completada sobre «{res_wiki['title']}»:**\n\n"
                    f"{res_wiki['summary']}\n\n"
                    f"📚 *Indexé {res_wiki['chunks_added']} fragmentos en mi memoria RAG, extraje {res_wiki['facts_extracted']} hechos a mi grafo y entrené mis neuronas (Loss: {res_wiki['loss']:.4f}). ¡Hazme cualquier pregunta sobre este tema!*"
                )
                return {
                    "reply": reply,
                    "confidence": 0.98,
                    "mode": "web_research",
                    "episode_id": res_wiki.get("episode_id"),
                    "loss": res_wiki.get("loss"),
                    "brain_state": self.get_brain_telemetry(),
                }
            return {
                "reply": str(res_wiki.get("message", "No pude investigar ese tema.")),
                "confidence": 0.5,
                "mode": "web_research",
                "brain_state": self.get_brain_telemetry(),
            }

        # 5. PILAR RAZONAMIENTO LÓGICO: ¿El usuario nos está enseñando una regla lógica ("Si X entonces Y")?
        logical_rule = NaturalChatParser.detect_logical_rule(user_message)
        if logical_rule is not None and not NaturalChatParser.is_question(user_message):
            cond, cons = logical_rule
            saved_rule = self.reasoning.add_rule(cond, cons)
            self.memory.log_self_teaching("logic_rule", f"Regla lógica #{saved_rule['id']}", f"Si {cond} → {cons}")
            self.save_state()
            return {
                "reply": f"🧩 **Regla condicional integrada en mi Motor Lógico:**\n`Si {cond} ➔ entonces {cons}`.\n¡Pregúntame qué pasa cuando ocurre eso o encadénala con más reglas!",
                "confidence": 0.98,
                "mode": "logical_reasoning",
                "brain_state": self.get_brain_telemetry(),
            }

        # 6. Si es una PREGUNTA: probar Razonamiento Lógico -> Base Documental RAG -> Grafo de Hechos
        is_q = NaturalChatParser.is_question(user_message)
        if is_q:
            # 6A. Motor de Inferencia Lógica Multi-Salto y Modus Ponens
            logic_ans = self.reasoning.answer_logical_question(user_message, self.memory.facts)
            if logic_ans is not None:
                self.last_user_stimulus = user_message
                self.last_ai_response = str(logic_ans["reply"])
                return {
                    "reply": logic_ans["reply"],
                    "confidence": 0.97,
                    "mode": "logical_reasoning",
                    "brain_state": self.get_brain_telemetry(),
                }

            # 6B. Grafo de Conocimiento Semántico directo (identidad del usuario, gustos, hechos exactos)
            fact_or_concept_ans = self.memory.query_facts_and_concepts(user_message, keywords)
            if fact_or_concept_ans is not None:
                reply_text, _ = fact_or_concept_ans
                self.last_user_stimulus = user_message
                self.last_ai_response = reply_text
                self.conversation_context = user_message
                return {
                    "reply": reply_text,
                    "confidence": 0.96,
                    "mode": "knowledge_graph",
                    "brain_state": self.get_brain_telemetry(),
                }

            # 6C. Búsqueda Vectorial en Documentos Subidos y Artículos Aprendidos (RAG)
            rag_hits = self.rag.search(user_message, self.tokenizer, self.nn, top_k=2, min_score=0.34)
            if rag_hits:
                top_hit = rag_hits[0]
                reply_rag = f"📚 **Según «{top_hit['title']}»** *(Similitud RAG: {int(top_hit['score']*100)}%)*:\n\n{top_hit['content']}"
                if len(rag_hits) > 1 and rag_hits[1]["score"] >= 0.45:
                    reply_rag += f"\n\n• *Fragmento relacionado:* {rag_hits[1]['content'][:220]}..."
                self.last_user_stimulus = user_message
                self.last_ai_response = reply_rag
                return {
                    "reply": reply_rag,
                    "confidence": top_hit["score"],
                    "mode": "document_rag",
                    "brain_state": self.get_brain_telemetry(),
                }

        # 7. Evaluar Micro-Transformer sobre Memoria Episódica y ejecutar Auto-Enseñanza
        extracted_fact = NaturalChatParser.extract_fact(user_message)
        saved_fact = None
        if extracted_fact is not None:
            saved_fact = self.memory.add_or_update_fact(
                extracted_fact["subject"],
                extracted_fact["relation"],
                extracted_fact["object"],
                category=extracted_fact["category"],
            )

        best_ep, confidence, breakdown, top_candidates = self.evaluate_candidates(
            user_message, context_text=self.conversation_context
        )

        auto_learned_events, _ = self.auto_learn_from_user_message(
            user_message, keywords, extracted_fact=extracted_fact
        )

        # 8. Si había un recuerdo episódico de alta confianza
        if best_ep is not None and confidence >= self.confidence_threshold and saved_fact is None:
            best_ep.uses += 1
            reply_text = best_ep.response
            stim_ids = self.tokenizer.encode(user_message, learn_vocab=False)
            resp_ids = self.tokenizer.encode(best_ep.response, learn_vocab=False)
            loss = self.nn.train_step(stim_ids, resp_ids, reward_weight=0.25)

            self.last_user_stimulus = user_message
            self.last_ai_response = reply_text
            self.last_episode_id = best_ep.id
            self.conversation_context = user_message
            self.save_state()
            return {
                "reply": reply_text,
                "confidence": round(confidence, 3),
                "mode": "neural_hybrid",
                "auto_learned_events": auto_learned_events,
                "episode_id": best_ep.id,
                "loss": round(loss, 5),
                "brain_state": self.get_brain_telemetry(),
            }

        # 9. AUTO-INVESTIGACIÓN AUTÓNOMA EN WIKIPEDIA cuando le haces una pregunta enciclopédica desconocida!
        if (
            is_q
            and self.auto_web_research
            and not self.curiosity_mode
            and re.match(r"^(?:que\s+es|quien\s+es|quien\s+fue|que\s+son|donde\s+esta|donde\s+queda|explicame\s+que\s+es|hablame\s+de)\s+(.+)$", norm_msg)
        ):
            res_wiki = self.research_topic_online(user_message)
            if res_wiki.get("ok"):
                reply = (
                    f"🌐 *No lo tenía en mi memoria, así que lo investigué sola en Wikipedia y lo aprendí ({res_wiki['chunks_added']} fragmentos RAG):*\n\n"
                    f"**{res_wiki['title']}:** {res_wiki['summary']}"
                )
                return {
                    "reply": reply,
                    "confidence": 0.95,
                    "mode": "web_research",
                    "episode_id": res_wiki.get("episode_id"),
                    "loss": res_wiki.get("loss"),
                    "brain_state": self.get_brain_telemetry(),
                }

        # 10. Si el Modo Pregunta Manual está activo
        if self.curiosity_mode and is_q:
            self.pending_unknown_stimulus = user_message
            self.save_state()
            return {
                "reply": f"🤔 Aún no conozco la respuesta a «{user_message}». ¿Qué debería responder? (O escribe «saltar»).",
                "confidence": round(confidence, 3),
                "mode": "curiosity_ask",
                "awaiting_teaching_for": user_message,
                "brain_state": self.get_brain_telemetry(),
            }

        # 11. Síntesis Autónoma + Auto-Enseñanza
        autonomous_reply = self.synthesize_autonomous_reply(
            user_message=user_message,
            keywords=keywords,
            saved_fact=saved_fact,
            best_ep=best_ep,
            confidence=confidence,
        )
        ep, step_loss = self.learn_association(
            stimulus=user_message,
            response=autonomous_reply,
            context=self.conversation_context,
            reward=1.1,
            source="auto_self_taught",
            steps=4,
        )

        self.last_user_stimulus = user_message
        self.last_ai_response = autonomous_reply
        self.last_episode_id = ep.id
        self.conversation_context = user_message
        self.save_state()

        return {
            "reply": autonomous_reply,
            "confidence": max(round(confidence, 3), 0.88 if auto_learned_events else 0.68),
            "mode": "auto_self_taught" if auto_learned_events else "autonomous_synthesis",
            "auto_learned_events": auto_learned_events,
            "episode_id": ep.id,
            "loss": round(step_loss, 5),
            "brain_state": self.get_brain_telemetry(),
        }

    # =========================================================================
    # SELF-PLAY, REFUERZO, ÉPOCAS Y TELEMETRÍA
    # =========================================================================
    def run_autonomous_self_play_step(self) -> Dict[str, object]:
        active_episodes = [ep for ep in self.memory.episodes if ep.reward > 0.25]
        if not active_episodes:
            return {"ok": False, "message": "Sin recuerdos suficientes."}

        sample_size = min(len(active_episodes), 6)
        sampled = random.sample(active_episodes, sample_size)
        total_loss = 0.0
        for ep in sampled:
            stim_ids = self.tokenizer.encode(ep.stimulus, learn_vocab=False)
            resp_ids = self.tokenizer.encode(ep.response, learn_vocab=False)
            neg_ids = None
            if len(active_episodes) > 1:
                neg_candidates = [e for e in active_episodes if e.id != ep.id]
                if neg_candidates:
                    neg_ids = self.tokenizer.encode(random.choice(neg_candidates).response, learn_vocab=False)
            total_loss += self.nn.train_step(stim_ids, resp_ids, negative_token_ids=neg_ids, reward_weight=ep.reward)

        avg_loss = total_loss / max(1, sample_size)
        self.nn.total_epochs_trained += 1
        self.memory.log_self_teaching(
            "replay",
            f"Auto-consolidación Transformer ({sample_size} recuerdos)",
            f"Ajustó pesos Q,K,V para «{sampled[0].stimulus}»",
            avg_loss,
        )
        self.save_state()
        return {"ok": True, "avg_loss": round(avg_loss, 5), "brain_state": self.get_brain_telemetry()}

    def apply_reinforcement(
        self,
        episode_id: Optional[int],
        feedback_type: str,
        corrected_response: Optional[str] = None,
    ) -> Dict[str, object]:
        target_ep = next((ep for ep in self.memory.episodes if ep.id == episode_id), None) if episode_id else None

        if feedback_type == "correct" and corrected_response:
            stimulus_to_use = target_ep.stimulus if target_ep else self.last_user_stimulus
            if not stimulus_to_use:
                return {"ok": False, "message": "No hay mensaje previo."}
            if target_ep:
                target_ep.reward = max(0.05, target_ep.reward - 0.6)
            new_ep, loss = self.learn_association(stimulus_to_use, corrected_response, reward=1.65, source="corrected", steps=8)
            self.last_episode_id = new_ep.id
            self.save_state()
            return {
                "ok": True,
                "action": "corrected",
                "episode": new_ep.to_dict(),
                "loss": round(loss, 5),
                "message": f"«{stimulus_to_use}» → «{corrected_response}» (Loss: {loss:.4f}).",
            }

        if not target_ep:
            return {"ok": False, "message": "Recuerdo no encontrado."}

        stim_ids = self.tokenizer.encode(target_ep.stimulus, learn_vocab=False)
        resp_ids = self.tokenizer.encode(target_ep.response, learn_vocab=False)
        if feedback_type == "reward":
            target_ep.reward = min(3.0, target_ep.reward + 0.35)
            loss = 0.0
            for _ in range(4):
                loss = self.nn.train_step(stim_ids, resp_ids, reward_weight=1.3)
            self.save_state()
            return {"ok": True, "action": "rewarded", "episode": target_ep.to_dict(), "loss": round(loss, 5)}

        if feedback_type == "punish":
            target_ep.reward = max(0.05, target_ep.reward - 0.55)
            loss = 0.0
            for _ in range(4):
                loss = self.nn.train_step(stim_ids, resp_ids, reward_weight=-1.0)
            self.save_state()
            return {"ok": True, "action": "punished", "episode": target_ep.to_dict(), "loss": round(loss, 5)}

        return {"ok": False, "message": "Feedback inválido."}

    def train_epochs(self, epochs: int = 10) -> Dict[str, object]:
        active_episodes = [ep for ep in self.memory.episodes if ep.reward > 0.2]
        if not active_episodes:
            return {"ok": False, "epochs_run": 0, "final_loss": 0.0}

        epoch_losses = []
        for _ in range(epochs):
            shuffled = list(active_episodes)
            random.shuffle(shuffled)
            batch_loss = 0.0
            for ep in shuffled:
                stim_ids = self.tokenizer.encode(ep.stimulus, learn_vocab=False)
                resp_ids = self.tokenizer.encode(ep.response, learn_vocab=False)
                neg_ids = None
                if len(active_episodes) > 1:
                    neg_candidates = [e for e in active_episodes if e.id != ep.id]
                    if neg_candidates:
                        neg_ids = self.tokenizer.encode(random.choice(neg_candidates).response, learn_vocab=False)
                batch_loss += self.nn.train_step(stim_ids, resp_ids, negative_token_ids=neg_ids, reward_weight=ep.reward)
            epoch_losses.append(round(batch_loss / max(1, len(shuffled)), 5))
            self.nn.total_epochs_trained += 1

        self.save_state()
        return {
            "ok": True,
            "epochs_run": epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "brain_state": self.get_brain_telemetry(),
        }

    def get_word_neighbors(self, word: str) -> List[Dict[str, float]]:
        norm_w = self.tokenizer.normalize(word)
        if not norm_w:
            return []
        tok = norm_w.split()[0]
        if tok not in self.tokenizer.word2idx:
            return []
        return self.nn.get_nearest_words(self.tokenizer.word2idx[tok], self.tokenizer.idx2word, top_k=6)

    def get_brain_telemetry(self) -> Dict[str, object]:
        vocab_words = [
            {"word": w, "freq": self.tokenizer.word_freq.get(w, 1)}
            for w in self.tokenizer.word2idx
            if not w.startswith("<")
        ]
        vocab_words.sort(key=lambda x: x["freq"], reverse=True)

        episodes_sorted = sorted([ep.to_dict() for ep in self.memory.episodes], key=lambda x: x["id"], reverse=True)
        rag_docs = self.rag.list_documents()
        tasks = self.toolkit.list_tasks()
        notes = self.toolkit.list_notes()

        return {
            "metrics": {
                "vocab_size": len(vocab_words),
                "episodes_count": len(self.memory.episodes),
                "facts_count": len(self.memory.facts) + len(self.reasoning.rules),
                "rag_docs_count": len(rag_docs),
                "tasks_pending": sum(1 for t in tasks if not t["done"]),
                "total_parameters": self.nn.parameter_count(),
                "epochs_trained": self.nn.total_epochs_trained,
                "steps_trained": self.nn.total_steps_trained,
                "last_loss": self.nn.loss_history[-1] if self.nn.loss_history else 0.0,
                "curiosity_mode": self.curiosity_mode,
                "auto_self_play": self.auto_self_play,
                "auto_web_research": self.auto_web_research,
                "awaiting_teaching_for": self.pending_unknown_stimulus,
            },
            "architecture": {
                "input_dim": self.nn.input_dim,
                "embed_dim": self.nn.embed_dim,
                "hidden1_dim": self.nn.hidden1_dim,
                "hidden2_dim": self.nn.hidden2_dim,
                "output_dim": self.nn.output_dim,
            },
            "activations": self.nn.last_activations,
            "token_attention": self.nn.last_token_attention,
            "loss_history": self.nn.loss_history[-60:],
            "episodes": episodes_sorted[:60],
            "facts": list(reversed(self.memory.facts)),
            "logic_rules": list(reversed(self.reasoning.rules)),
            "rag_docs": rag_docs,
            "tasks": tasks,
            "notes": notes,
            "self_teaching_log": list(reversed(self.memory.self_teaching_log[-40:])),
            "top_vocab": vocab_words[:50],
        }

    def save_state(self):
        self.nn.save_weights(self.weights_path)
        payload = {
            "tokenizer": self.tokenizer.to_dict(),
            "memory": self.memory.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "settings": {
                "curiosity_mode": self.curiosity_mode,
                "auto_self_play": self.auto_self_play,
                "auto_web_research": self.auto_web_research,
                "confidence_threshold": self.confidence_threshold,
            },
        }
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_state(self):
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.tokenizer.from_dict(payload.get("tokenizer", {}))
            self.memory.from_dict(payload.get("memory", {}))
            self.reasoning.from_dict(payload.get("reasoning", {}))
            settings = payload.get("settings", {})
            self.curiosity_mode = bool(settings.get("curiosity_mode", False))
            self.auto_self_play = bool(settings.get("auto_self_play", True))
            self.auto_web_research = bool(settings.get("auto_web_research", True))
            self.confidence_threshold = float(settings.get("confidence_threshold", 0.44))
            self.nn.load_weights(self.weights_path)
            self.nn.expand_vocab(len(self.tokenizer.word2idx))
        except Exception as e:
            print(f"[Aviso] Inicializando nuevo cerebro Micro-Transformer v2 ({e})...")
            self.seed_initial_brain()
