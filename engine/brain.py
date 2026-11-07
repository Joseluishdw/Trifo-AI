"""
NeuroZero 3.0 — Orquestador Cognitivo Neuro-Simbólico Thread-Safe.
Integra los 4 Pilares desde cero (Python stdlib + NumPy):
1. Micro-Transformer Multi-Cabeza (MHSA H=4) + LayerNorm + Global L2 Grad Clip + Context Backprop.
2. Memoria de Trabajo Multi-Turno (Buffer Circular con decaimiento exponencial) + Neural-Markov Reranker (K=5) + RAG con Caché Vectorial.
3. Motor de Razonamiento Simbólico con Detección de Contradicciones Lógicas y Backward Chaining.
4. Agente Multi-Herramienta Encadenado, Ecuaciones Cuadráticas (ℝ/ℂ), Estadística CSV en NumPy y Respaldos Completos.
"""

import io
import os
import re
import math
import time
import json
import base64
import random
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tokenizer import ScratchTokenizer, NaturalChatParser, strip_accents, STOPWORDS_ES
from .neural_net import ScratchNeuralNetwork, atomic_replace_file
from .memory import HybridMemorySystem, EpisodicMemoryItem
from .reasoning import LogicalReasoningEngine
from .knowledge_rag import ScratchVectorRAG, DocumentParserFromScratch, WikipediaAutoResearcher
from .tools_agent import ScratchAgentToolkit


INITIAL_SEED_KNOWLEDGE = [
    (
        "hola",
        "¡Hola! Soy NeuroZero 3.0 (Trifo-AI). Cuento con Micro-Transformer Multi-Cabeza (4 cabezas + LayerNorm), "
        "Memoria Multi-Turno, Detección de Contradicciones, Backward Chaining, RAG con caché vectorial y Agente Multi-Herramienta. ¿Qué construimos o exploramos hoy?",
    ),
    (
        "buenos dias",
        "¡Buenos días! Mis 4 cabezas de atención y herramientas están activas. ¿Qué proyecto, ecuación, documento o deducción trabajamos hoy?",
    ),
    (
        "buenas tardes",
        "¡Buenas tardes! Puedes conversar conmigo, subirme archivos (.csv, .pdf, .docx, .md, .txt), encadenar herramientas o probar razonamiento lógico.",
    ),
    (
        "buenas noches",
        "¡Buenas noches! Mi memoria neuronal y simbólica sigue lista para entrenar o asistirte.",
    ),
    (
        "como estas",
        "¡Excelente! Operando con arquitectura NeuroZero 3.0 en NumPy puro y sincronización atómica. ¿Y tú cómo estás?",
    ),
    (
        "que puedes hacer",
        "En NeuroZero 3.0 puedo: 1) Aprender con mi Micro-Transformer de 4 cabezas y memoria multi-turno, "
        "2) Indexar documentos RAG con caché vectorial y analizar estadísticas de archivos .csv, "
        "3) Detectar contradicciones lógicas y explicar por qué algo es cierto con Backward Chaining, y "
        "4) Encadenar múltiples herramientas (ecuaciones cuadráticas reales/complejas, cálculos AST, notas, tareas y archivos).",
    ),
    (
        "quien eres",
        "Soy NeuroZero 3.0 (Trifo-AI), un motor neuro-simbólico construido 100% desde cero con la librería estándar de Python y NumPy.",
    ),
    ("gracias", "¡Con gusto! Cada conversación ajusta mis cabezas de atención y fortalece mi grafo."),
    ("adios", "¡Hasta pronto! Todos mis pesos v3, recuerdos y documentos quedan respaldados atómicamente en disco."),
]

LEGACY_FILES = (
    "brain_memory.json",
    "brain_weights.npz",
    "brain_memory_v2.json",
    "brain_weights_v2.npz",
)


class ScratchAIEngine:
    def __init__(self, data_dir: str):
        self._lock = threading.RLock()
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.exports_dir = os.path.join(os.path.dirname(self.data_dir), "exports")
        os.makedirs(self.exports_dir, exist_ok=True)

        self._cleanup_legacy_files()

        self.weights_path = os.path.join(self.data_dir, "brain_weights_v3.npz")
        self.memory_path = os.path.join(self.data_dir, "brain_memory_v3.json")
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
            num_heads=4,
            hidden1_dim=128,
            hidden2_dim=96,
            learning_rate=0.005,
            max_grad_norm=3.0,
        )

        self.curiosity_mode: bool = False
        self.auto_self_play: bool = True
        self.auto_web_research: bool = True
        self.confidence_threshold: float = 0.44

        self.pending_unknown_stimulus: Optional[str] = None
        self.last_user_stimulus: str = ""
        self.last_ai_response: str = ""
        self.last_ai_question_concept: Optional[str] = None
        self.last_episode_id: Optional[int] = None

        if os.path.exists(self.memory_path) and os.path.exists(self.weights_path):
            self.load_state()
        else:
            self.seed_initial_brain()

    def _cleanup_legacy_files(self):
        for fname in LEGACY_FILES:
            fpath = os.path.join(self.data_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    def seed_initial_brain(self):
        with self._lock:
            self.memory.add_or_update_fact("ia", "nombre", "NeuroZero 3.0", category="ai")
            self.memory.add_or_update_fact(
                "ia",
                "identidad",
                "un motor neuro-simbólico v3 con Micro-Transformer Multi-Cabeza (H=4), LayerNorm, RAG con caché vectorial, Backward Chaining y Agente Multi-Herramienta desde cero",
                category="ai",
            )
            for stim, resp in INITIAL_SEED_KNOWLEDGE:
                self._register_and_encode(stim)
                self._register_and_encode(resp)
                self.memory.add_episode(stimulus=stim, response=resp, context="", reward=1.25, source="seed")

            self.memory.log_self_teaching(
                kind="init",
                summary="NeuroZero 3.0 Inicializado",
                detail="MHSA (4 cabezas) + LayerNorm + WorkingMemory(4) + Lógica con Contradicciones + Agente v3 listos.",
            )
            self.train_epochs(epochs=8)
            self.save_state()

    def reset_brain(self, include_seed: bool = True):
        with self._lock:
            self.tokenizer = ScratchTokenizer()
            self.memory = HybridMemorySystem()
            self.reasoning = LogicalReasoningEngine()
            self.rag.clear_all()
            self.toolkit.clear_all()
            self.nn = ScratchNeuralNetwork(
                vocab_size=len(self.tokenizer.word2idx),
                embed_dim=64,
                num_heads=4,
                hidden1_dim=128,
                hidden2_dim=96,
                learning_rate=0.005,
                max_grad_norm=3.0,
            )
            self.pending_unknown_stimulus = None
            self.last_user_stimulus = ""
            self.last_ai_response = ""
            self.last_ai_question_concept = None
            self.last_episode_id = None

            if include_seed:
                self.seed_initial_brain()
            else:
                self.memory.log_self_teaching(
                    kind="reset",
                    summary="Tabula Rasa Completa (v3)",
                    detail="Pesos MHSA, memoria multi-turno, reglas y biblioteca RAG reiniciados desde cero.",
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

    def _encode_context_string_as_turns(self, context_str: str) -> List[Tuple[List[int], float]]:
        if not context_str or not context_str.strip():
            return []
        parts = [p.strip() for p in context_str.split(" | ") if p.strip()][-4:]
        if not parts:
            return []
        n = len(parts)
        decay = self.memory.working_memory.decay_rate
        raw_w = [math.exp(-decay * (n - 1 - i)) for i in range(n)]
        total_w = sum(raw_w) or 1.0
        turns: List[Tuple[List[int], float]] = []
        for i, part in enumerate(parts):
            t_ids = self.tokenizer.encode(part, learn_vocab=False)
            if t_ids:
                turns.append((t_ids, raw_w[i] / total_w))
        return turns

    def ingest_document_bytes(self, filename: str, raw_bytes: bytes) -> Dict[str, object]:
        with self._lock:
            csv_stats = None
            csv_report = None
            if filename.lower().endswith(".csv"):
                csv_stats = self.toolkit.register_csv_dataset(filename, raw_bytes)
                if csv_stats:
                    csv_report = self.toolkit.format_csv_stats_report(filename, csv_stats)

            text = DocumentParserFromScratch.parse_bytes(filename, raw_bytes)
            if not text or len(text.strip()) < 10:
                return {
                    "ok": False,
                    "message": f"No se pudo extraer texto legible de `{filename}`.",
                }

            res = self.ingest_text_document(title=filename, text=text, source_type="file")
            if csv_report and res.get("ok"):
                res["csv_stats"] = csv_stats
                res["csv_report"] = csv_report
            return res

    def ingest_text_document(self, title: str, text: str, source_type: str = "file") -> Dict[str, object]:
        with self._lock:
            for sample_line in text.splitlines()[:40]:
                if sample_line.strip():
                    self._register_and_encode(sample_line[:280], record_attention=False)

            res = self.rag.ingest_document(
                title=title,
                text=text,
                source_type=source_type,
                tokenizer=self.tokenizer,
                neural_net=self.nn,
            )
            if not res.get("ok"):
                return {"ok": False, "message": "El texto es demasiado corto para indexarse."}

            chunks: List[str] = res.get("chunks", [])  # type: ignore
            facts_extracted = 0
            last_loss = self.nn.loss_history[-1] if self.nn.loss_history else 0.0

            for chunk in chunks[:25]:
                self._register_and_encode(chunk, record_attention=False)
                sentences = re.split(r"(?<=[.!?])\s+", chunk)
                for sent in sentences[:4]:
                    sent_clean = sent.strip()
                    if 15 <= len(sent_clean) <= 220:
                        fact = NaturalChatParser.extract_fact(sent_clean)
                        if fact and fact["category"] == "world":
                            conflict = self.reasoning.check_fact_contradiction(fact, self.memory.facts)
                            if conflict is None:
                                self.memory.add_or_update_fact(
                                    fact["subject"], fact["relation"], fact["object"], category="world"
                                )
                                facts_extracted += 1

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

            self.memory.log_self_teaching(
                kind="rag_ingest",
                summary=f"Estudió «{title}» ({source_type})",
                detail=f"{len(chunks)} fragmentos RAG en caché vectorial, +{facts_extracted} hechos al grafo.",
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
        wiki_data = WikipediaAutoResearcher.search_and_fetch(query)
        if not wiki_data:
            return {
                "ok": False,
                "message": "No pude encontrar un artículo en Wikipedia para ese término (o no hay conexión a Internet).",
            }

        with self._lock:
            ingested = self.ingest_text_document(
                title=f"Wiki: {wiki_data['title']}",
                text=wiki_data["content"],
                source_type="wikipedia",
            )
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
        self, user_text: str
    ) -> Tuple[Optional[EpisodicMemoryItem], float, Dict[str, float], List[Dict[str, object]], np.ndarray]:
        tokens_str = self.tokenizer.tokenize(user_text, learn_vocab=False)
        msg_ids = self.tokenizer.encode(user_text, learn_vocab=False)
        ctx_turns = self.memory.working_memory.get_encoded_turns(self.tokenizer)

        msg_vec = self.nn.encode_tokens(msg_ids, tokens_str=tokens_str, record_attention=True)
        ctx_vec, _ = self.nn.encode_context(ctx_turns=ctx_turns)

        pred_resp_vec, _ = self.nn.forward(msg_vec, ctx_vec, record_telemetry=True)

        if not self.memory.episodes:
            return None, 0.0, {"neural": 0.0, "embedding": 0.0, "lexical": 0.0}, [], pred_resp_vec

        wm_summary = self.memory.working_memory.as_summary_string()
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
            if wm_summary and ep.context:
                ctx_bonus = 0.08 * self._lexical_overlap_score(wm_summary, ep.context)

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
            return None, 0.0, {"neural": 0.0, "embedding": 0.0, "lexical": 0.0}, [], pred_resp_vec

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
        return best["episode"], best["score"], best["breakdown"], top_list, pred_resp_vec

    def learn_association(
        self,
        stimulus: str,
        response: str,
        context: Optional[str] = None,
        reward: float = 1.25,
        source: str = "auto_self_taught",
        steps: int = 5,
    ) -> Tuple[EpisodicMemoryItem, float]:
        with self._lock:
            stim_ids = self._register_and_encode(stimulus)
            resp_ids = self._register_and_encode(response)

            if context is not None:
                if context.strip():
                    self._register_and_encode(context)
                ctx_str = context.strip()
                ctx_turns = self._encode_context_string_as_turns(ctx_str)
            else:
                ctx_str = self.memory.working_memory.as_summary_string()
                ctx_turns = self.memory.working_memory.get_encoded_turns(self.tokenizer)

            episode = self.memory.add_episode(
                stimulus=stimulus,
                response=response,
                context=ctx_str,
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
                    ctx_turns=ctx_turns,
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
        pred_resp_vec: np.ndarray,
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
                deductions = self.reasoning.deduce_all_about_entity(subj, self.memory.facts)
                ded_note = ""
                if deductions:
                    ded_note = f"\n🧩 *¡Además deduje por inferencia lógica que: **{deductions[0]['conclusion']}**!*"
                self.last_ai_question_concept = subj.lower()
                return (
                    f"¡Aprendido! Guardé en mi grafo que «{subj} {rel} {obj}».{ded_note}\n"
                    f"¿Qué más sabes o quieres explorar sobre {subj.lower()}?"
                )

        if is_q:
            if keywords:
                markov_reranked = self.memory.generate_sentence_from_seed(
                    keywords,
                    max_len=15,
                    num_candidates=5,
                    tokenizer=self.tokenizer,
                    neural_net=self.nn,
                    target_latent_vec=pred_resp_vec,
                )
                if markov_reranked:
                    return f"Por lo que he aprendido sobre «{keywords[0]}»: {markov_reranked}."

                for kw in keywords:
                    related = self.memory.get_related_concepts(kw, top_k=3)
                    if related:
                        self.last_ai_question_concept = kw
                        return (
                            f"Tengo «{kw}» vinculado con {', '.join(related)}, pero aún puedo profundizar más. "
                            f"Cuéntame o dime *«Investiga sobre {kw}»*."
                        )

            if best_ep is not None and confidence >= 0.25:
                return best_ep.response

            main_kw = keywords[0] if keywords else user_message
            self.last_ai_question_concept = main_kw
            return (
                f"Aún no tengo datos sobre «{main_kw}». Platícame al respecto, súbeme un documento con 📎 "
                f"o dime *«Investiga sobre {main_kw}»*."
            )

        main_kw = keywords[0] if keywords else ""
        if main_kw:
            self.last_ai_question_concept = main_kw
            related = self.memory.get_related_concepts(main_kw, top_k=2)
            rel_note = f" (vinculado en mis neuronas con {', '.join(related)})" if related else ""
            return f"¡Interesante! Ya absorbí en mi red que {shifted_msg}{rel_note}. ¿Qué más me cuentas sobre {main_kw}?"

        return f"He procesado «{user_message}» en mis 4 cabezas de atención. ¿En qué más trabajamos?"

    def _finalize_turn_context(self, user_message: str, ai_reply: str, episode_id: Optional[int] = None):
        self.last_user_stimulus = user_message
        self.last_ai_response = ai_reply
        if episode_id is not None:
            self.last_episode_id = episode_id
        self.memory.working_memory.push_turn(user_message)

    def process_chat_message(self, user_message: str) -> Dict[str, object]:
        with self._lock:
            user_message = user_message.strip()
            if not user_message:
                return {"reply": "Escribe un mensaje, ecuación, tarea o sube un documento.", "confidence": 0.0}

            self._register_and_encode(user_message, record_attention=True)
            keywords = self.tokenizer.extract_keywords(user_message)
            norm_msg = strip_accents(user_message.lower())

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
                self._finalize_turn_context(learned_stim, user_message, ep.id)
                self.save_state()
                return {
                    "reply": f"🧠 ¡Aprendido! Cuando digas «{learned_stim}», responderé «{user_message}» (Loss: {loss:.4f}).",
                    "confidence": 0.99,
                    "mode": "learned_from_curiosity",
                    "episode_id": ep.id,
                    "loss": round(loss, 5),
                    "brain_state": self.get_brain_telemetry(),
                }

            explicit_pair = NaturalChatParser.detect_explicit_teaching(user_message)
            if explicit_pair is not None:
                trig, resp = explicit_pair
                ep, loss = self.learn_association(trig, resp, reward=1.5, source="chat_taught", steps=8)
                self._finalize_turn_context(trig, resp, ep.id)
                self.save_state()
                return {
                    "reply": f"⚡ ¡Regla entrenada en el Micro-Transformer MHSA! «{trig}» → «{resp}» (Loss: {loss:.4f}).",
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

            tool_result = self.toolkit.try_execute_tool(user_message)
            if tool_result is not None:
                msg_vec = self.nn.encode_tokens(
                    self.tokenizer.encode(user_message),
                    tokens_str=self.tokenizer.tokenize(user_message),
                    record_attention=True,
                )
                ctx_vec, _ = self.nn.encode_context(ctx_turns=self.memory.working_memory.get_encoded_turns(self.tokenizer))
                self.nn.forward(msg_vec, ctx_vec, record_telemetry=True)
                reply_str = str(tool_result["reply"])
                self._finalize_turn_context(user_message, reply_str)
                self.save_state()
                return {
                    "reply": reply_str,
                    "confidence": 1.0,
                    "mode": "agent_tool",
                    "tool_used": tool_result["tool_name"],
                    "brain_state": self.get_brain_telemetry(),
                }

            if re.match(r"^(?:investiga(?:\s+sobre)?|busca\s+en\s+wikipedia(?:\s+sobre)?|estudia\s+sobre)\s+(.+)$", norm_msg):
                res_wiki = self.research_topic_online(user_message)
                if res_wiki.get("ok"):
                    reply = (
                        f"🌐 **Auto-Investigación completada sobre «{res_wiki['title']}»:**\n\n"
                        f"{res_wiki['summary']}\n\n"
                        f"📚 *Indexé {res_wiki['chunks_added']} fragmentos en caché vectorial RAG, extraje {res_wiki['facts_extracted']} hechos y entrené mis cabezas de atención (Loss: {res_wiki['loss']:.4f}).*"
                    )
                    self._finalize_turn_context(user_message, reply, res_wiki.get("episode_id"))  # type: ignore
                    self.save_state()
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

            logical_rule = NaturalChatParser.detect_logical_rule(user_message)
            if logical_rule is not None and not NaturalChatParser.is_question(user_message):
                cond, cons = logical_rule
                rule_conflict = self.reasoning.check_rule_contradiction(cond, cons)
                if rule_conflict is not None:
                    warn_reply = (
                        f"{rule_conflict['message']}\n\n"
                        f"🛡️ *No inserté la regla contradictoria para proteger la consistencia formal de mi motor lógico.*"
                    )
                    self.memory.log_self_teaching("contradiction", "Contradicción en regla bloqueada", f"Si {cond} → {cons}")
                    return {
                        "reply": warn_reply,
                        "confidence": 0.99,
                        "mode": "logical_contradiction",
                        "brain_state": self.get_brain_telemetry(),
                    }

                saved_rule = self.reasoning.add_rule(cond, cons)
                self.memory.log_self_teaching("logic_rule", f"Regla lógica #{saved_rule['id']}", f"Si {cond} → {cons}")
                reply_rule = (
                    f"🧩 **Regla condicional integrada en mi Motor Lógico:**\n"
                    f"`Si {cond} ➔ entonces {cons}`.\n"
                    f"¡Pregúntame qué pasa si ocurre eso o por qué sucede su consecuencia!"
                )
                self._finalize_turn_context(user_message, reply_rule)
                self.save_state()
                return {
                    "reply": reply_rule,
                    "confidence": 0.98,
                    "mode": "logical_reasoning",
                    "brain_state": self.get_brain_telemetry(),
                }

            is_q = NaturalChatParser.is_question(user_message)
            if is_q:
                logic_ans = self.reasoning.answer_logical_question(user_message, self.memory.facts)
                if logic_ans is not None:
                    reply_logic = str(logic_ans["reply"])
                    self._finalize_turn_context(user_message, reply_logic)
                    self.save_state()
                    return {
                        "reply": reply_logic,
                        "confidence": 0.98,
                        "mode": "logical_reasoning",
                        "brain_state": self.get_brain_telemetry(),
                    }

                fact_or_concept_ans = self.memory.query_facts_and_concepts(user_message, keywords)
                if fact_or_concept_ans is not None:
                    reply_text, _ = fact_or_concept_ans
                    self._finalize_turn_context(user_message, reply_text)
                    self.save_state()
                    return {
                        "reply": reply_text,
                        "confidence": 0.96,
                        "mode": "knowledge_graph",
                        "brain_state": self.get_brain_telemetry(),
                    }

                rag_hits = self.rag.search(user_message, self.tokenizer, self.nn, top_k=2, min_score=0.34)
                if rag_hits:
                    top_hit = rag_hits[0]
                    reply_rag = (
                        f"📚 **Según «{top_hit['title']}»** *(Similitud RAG: {int(float(top_hit['score'])*100)}%)*:\n\n"
                        f"{top_hit['content']}"
                    )
                    if len(rag_hits) > 1 and float(rag_hits[1]["score"]) >= 0.45:
                        reply_rag += f"\n\n• *Fragmento relacionado:* {str(rag_hits[1]['content'])[:220]}..."
                    self._finalize_turn_context(user_message, reply_rag)
                    self.save_state()
                    return {
                        "reply": reply_rag,
                        "confidence": top_hit["score"],
                        "mode": "document_rag",
                        "brain_state": self.get_brain_telemetry(),
                    }

            extracted_fact = NaturalChatParser.extract_fact(user_message)
            saved_fact = None
            if extracted_fact is not None:
                if extracted_fact["category"] == "world":
                    fact_conflict = self.reasoning.check_fact_contradiction(extracted_fact, self.memory.facts)
                    if fact_conflict is not None:
                        warn_reply = (
                            f"{fact_conflict['message']}\n\n"
                            f"🛡️ *He bloqueado este axioma para evitar inconsistencias en mi Grafo de Conocimiento.*"
                        )
                        self.memory.log_self_teaching(
                            "contradiction",
                            f"Conflicto lógico detectado sobre «{extracted_fact['subject']}»",
                            f"{extracted_fact['subject']} {extracted_fact['relation']} {extracted_fact['object']}",
                        )
                        self.save_state()
                        return {
                            "reply": warn_reply,
                            "confidence": 0.99,
                            "mode": "logical_contradiction",
                            "brain_state": self.get_brain_telemetry(),
                        }

                saved_fact = self.memory.add_or_update_fact(
                    extracted_fact["subject"],
                    extracted_fact["relation"],
                    extracted_fact["object"],
                    category=extracted_fact["category"],
                )

            best_ep, confidence, _, _, pred_resp_vec = self.evaluate_candidates(user_message)

            auto_learned_events, _ = self.auto_learn_from_user_message(
                user_message, keywords, extracted_fact=extracted_fact
            )

            if best_ep is not None and confidence >= self.confidence_threshold and saved_fact is None:
                best_ep.uses += 1
                reply_text = best_ep.response
                stim_ids = self.tokenizer.encode(user_message, learn_vocab=False)
                resp_ids = self.tokenizer.encode(best_ep.response, learn_vocab=False)
                ctx_turns = self.memory.working_memory.get_encoded_turns(self.tokenizer)
                loss = self.nn.train_step(stim_ids, resp_ids, ctx_turns=ctx_turns, reward_weight=0.25)

                self._finalize_turn_context(user_message, reply_text, best_ep.id)
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

            if (
                is_q
                and self.auto_web_research
                and not self.curiosity_mode
                and re.match(
                    r"^(?:que\s+es|quien\s+es|quien\s+fue|que\s+son|donde\s+esta|donde\s+queda|explicame\s+que\s+es|hablame\s+de)\s+(.+)$",
                    norm_msg,
                )
            ):
                res_wiki = self.research_topic_online(user_message)
                if res_wiki.get("ok"):
                    reply = (
                        f"🌐 *No lo tenía en mi memoria, así que lo investigué sola en Wikipedia y lo aprendí ({res_wiki['chunks_added']} fragmentos RAG):*\n\n"
                        f"**{res_wiki['title']}:** {res_wiki['summary']}"
                    )
                    self._finalize_turn_context(user_message, reply, res_wiki.get("episode_id"))  # type: ignore
                    self.save_state()
                    return {
                        "reply": reply,
                        "confidence": 0.95,
                        "mode": "web_research",
                        "episode_id": res_wiki.get("episode_id"),
                        "loss": res_wiki.get("loss"),
                        "brain_state": self.get_brain_telemetry(),
                    }

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

            autonomous_reply = self.synthesize_autonomous_reply(
                user_message=user_message,
                keywords=keywords,
                saved_fact=saved_fact,
                best_ep=best_ep,
                confidence=confidence,
                pred_resp_vec=pred_resp_vec,
            )
            ep, step_loss = self.learn_association(
                stimulus=user_message,
                response=autonomous_reply,
                reward=1.1,
                source="auto_self_taught",
                steps=4,
            )

            self._finalize_turn_context(user_message, autonomous_reply, ep.id)
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

    def run_autonomous_self_play_step(self) -> Dict[str, object]:
        with self._lock:
            active_episodes = [ep for ep in self.memory.episodes if ep.reward > 0.25]
            if not active_episodes:
                return {"ok": False, "message": "Sin recuerdos suficientes."}

            sample_size = min(len(active_episodes), 6)
            sampled = random.sample(active_episodes, sample_size)
            total_loss = 0.0
            for ep in sampled:
                stim_ids = self.tokenizer.encode(ep.stimulus, learn_vocab=False)
                resp_ids = self.tokenizer.encode(ep.response, learn_vocab=False)
                ctx_turns = self._encode_context_string_as_turns(ep.context)
                neg_ids = None
                if len(active_episodes) > 1:
                    neg_candidates = [e for e in active_episodes if e.id != ep.id]
                    if neg_candidates:
                        neg_ids = self.tokenizer.encode(random.choice(neg_candidates).response, learn_vocab=False)
                total_loss += self.nn.train_step(
                    stim_ids,
                    resp_ids,
                    ctx_turns=ctx_turns,
                    negative_token_ids=neg_ids,
                    reward_weight=ep.reward,
                )

            avg_loss = total_loss / max(1, sample_size)
            self.nn.total_epochs_trained += 1
            self.rag.invalidate_vector_cache()
            self.memory.log_self_teaching(
                "replay",
                f"Auto-consolidación MHSA ({sample_size} recuerdos)",
                f"Ajustó 4 cabezas + LayerNorm para «{sampled[0].stimulus}»",
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
        with self._lock:
            target_ep = next((ep for ep in self.memory.episodes if ep.id == episode_id), None) if episode_id else None

            if feedback_type == "correct" and corrected_response:
                stimulus_to_use = target_ep.stimulus if target_ep else self.last_user_stimulus
                if not stimulus_to_use:
                    return {"ok": False, "message": "No hay mensaje previo."}
                if target_ep:
                    target_ep.reward = max(0.05, target_ep.reward - 0.6)
                new_ep, loss = self.learn_association(
                    stimulus_to_use, corrected_response, reward=1.65, source="corrected", steps=8
                )
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
            ctx_turns = self._encode_context_string_as_turns(target_ep.context)

            if feedback_type == "reward":
                target_ep.reward = min(3.0, target_ep.reward + 0.35)
                loss = 0.0
                for _ in range(4):
                    loss = self.nn.train_step(stim_ids, resp_ids, ctx_turns=ctx_turns, reward_weight=1.3)
                self.save_state()
                return {"ok": True, "action": "rewarded", "episode": target_ep.to_dict(), "loss": round(loss, 5)}

            if feedback_type == "punish":
                target_ep.reward = max(0.05, target_ep.reward - 0.55)
                loss = 0.0
                for _ in range(4):
                    loss = self.nn.train_step(stim_ids, resp_ids, ctx_turns=ctx_turns, reward_weight=-1.0)
                self.save_state()
                return {"ok": True, "action": "punished", "episode": target_ep.to_dict(), "loss": round(loss, 5)}

            return {"ok": False, "message": "Feedback inválido."}

    def train_epochs(self, epochs: int = 10) -> Dict[str, object]:
        with self._lock:
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
                    ctx_turns = self._encode_context_string_as_turns(ep.context)
                    neg_ids = None
                    if len(active_episodes) > 1:
                        neg_candidates = [e for e in active_episodes if e.id != ep.id]
                        if neg_candidates:
                            neg_ids = self.tokenizer.encode(random.choice(neg_candidates).response, learn_vocab=False)
                    batch_loss += self.nn.train_step(
                        stim_ids,
                        resp_ids,
                        ctx_turns=ctx_turns,
                        negative_token_ids=neg_ids,
                        reward_weight=ep.reward,
                    )
                epoch_losses.append(round(batch_loss / max(1, len(shuffled)), 5))
                self.nn.total_epochs_trained += 1

            self.rag.invalidate_vector_cache()
            self.save_state()
            return {
                "ok": True,
                "epochs_run": epochs,
                "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
                "brain_state": self.get_brain_telemetry(),
            }

    def update_settings(self, body: dict) -> Dict[str, object]:
        with self._lock:
            if "curiosity_mode" in body:
                self.curiosity_mode = bool(body["curiosity_mode"])
                if not self.curiosity_mode:
                    self.pending_unknown_stimulus = None
            if "auto_self_play" in body:
                self.auto_self_play = bool(body["auto_self_play"])
            if "auto_web_research" in body:
                self.auto_web_research = bool(body["auto_web_research"])
            self.save_state()
            return {"ok": True, "brain_state": self.get_brain_telemetry()}

    def delete_episode(self, ep_id: int) -> Dict[str, object]:
        with self._lock:
            self.memory.episodes = [ep for ep in self.memory.episodes if ep.id != ep_id]
            self.save_state()
            return {"ok": True, "brain_state": self.get_brain_telemetry()}

    def delete_fact(self, fact_id: int) -> Dict[str, object]:
        with self._lock:
            self.memory.facts = [f for f in self.memory.facts if int(f.get("id", 0)) != fact_id]
            self.save_state()
            return {"ok": True, "brain_state": self.get_brain_telemetry()}

    def delete_rag_document(self, doc_id: int) -> Dict[str, object]:
        with self._lock:
            self.rag.delete_document(doc_id)
            return {"ok": True, "brain_state": self.get_brain_telemetry()}

    def get_word_neighbors(self, word: str) -> List[Dict[str, float]]:
        with self._lock:
            norm_w = self.tokenizer.normalize(word)
            if not norm_w:
                return []
            tok = norm_w.split()[0]
            if tok not in self.tokenizer.word2idx:
                return []
            return self.nn.get_nearest_words(self.tokenizer.word2idx[tok], self.tokenizer.idx2word, top_k=6)

    def get_brain_telemetry(self) -> Dict[str, object]:
        with self._lock:
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
                    "grad_norm": self.nn.last_grad_norm,
                    "curiosity_mode": self.curiosity_mode,
                    "auto_self_play": self.auto_self_play,
                    "auto_web_research": self.auto_web_research,
                    "awaiting_teaching_for": self.pending_unknown_stimulus,
                },
                "architecture": {
                    "input_dim": self.nn.input_dim,
                    "embed_dim": self.nn.embed_dim,
                    "num_heads": self.nn.num_heads,
                    "head_dim": self.nn.head_dim,
                    "hidden1_dim": self.nn.hidden1_dim,
                    "hidden2_dim": self.nn.hidden2_dim,
                    "output_dim": self.nn.output_dim,
                },
                "activations": self.nn.last_activations,
                "token_attention": self.nn.last_token_attention,
                "head_attention": self.nn.last_head_attention,
                "working_memory": [
                    {"turn": t, "weight": w} for t, w in self.memory.working_memory.get_weighted_turns()
                ],
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
        with self._lock:
            self.nn.save_weights(self.weights_path)
            payload = {
                "version": "NeuroZero-3.0",
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
            tmp_memory_path = f"{self.memory_path}.tmp"
            with open(tmp_memory_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace_file(tmp_memory_path, self.memory_path)

    def load_state(self):
        with self._lock:
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
                self.rag.invalidate_vector_cache()
            except Exception as e:
                print(f"[Aviso] Inicializando nuevo cerebro Micro-Transformer MHSA v3 ({e})...")
                self.seed_initial_brain()

    def export_full_backup(self) -> Dict[str, object]:
        with self._lock:
            self.save_state()
            with open(self.weights_path, "rb") as wf:
                weights_b64 = base64.b64encode(wf.read()).decode("ascii")

            return {
                "version": "NeuroZero-3.0",
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tokenizer": self.tokenizer.to_dict(),
                "memory": self.memory.to_dict(),
                "reasoning": self.reasoning.to_dict(),
                "settings": {
                    "curiosity_mode": self.curiosity_mode,
                    "auto_self_play": self.auto_self_play,
                    "auto_web_research": self.auto_web_research,
                    "confidence_threshold": self.confidence_threshold,
                },
                "rag_state": self.rag.export_state_dict(),
                "toolkit_state": self.toolkit.export_state_dict(),
                "weights_npz_base64": weights_b64,
            }

    def import_full_backup(self, backup_data: dict) -> Dict[str, object]:
        with self._lock:
            if not isinstance(backup_data, dict) or "weights_npz_base64" not in backup_data or "tokenizer" not in backup_data:
                return {"ok": False, "message": "El archivo no es un respaldo válido de NeuroZero 3.0."}

            try:
                raw_npz = base64.b64decode(str(backup_data["weights_npz_base64"]))
                tmp_npz = f"{self.weights_path}.import.tmp"
                with open(tmp_npz, "wb") as f:
                    f.write(raw_npz)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace_file(tmp_npz, self.weights_path)

                self.nn.load_weights(self.weights_path)
                self.tokenizer.from_dict(backup_data.get("tokenizer", {}))
                self.nn.expand_vocab(len(self.tokenizer.word2idx))
                self.memory.from_dict(backup_data.get("memory", {}))
                self.reasoning.from_dict(backup_data.get("reasoning", {}))

                settings = backup_data.get("settings", {})
                self.curiosity_mode = bool(settings.get("curiosity_mode", False))
                self.auto_self_play = bool(settings.get("auto_self_play", True))
                self.auto_web_research = bool(settings.get("auto_web_research", True))
                self.confidence_threshold = float(settings.get("confidence_threshold", 0.44))

                if "rag_state" in backup_data:
                    self.rag.import_state_dict(backup_data["rag_state"])
                if "toolkit_state" in backup_data:
                    self.toolkit.import_state_dict(backup_data["toolkit_state"])

                self.pending_unknown_stimulus = None
                self.rag.invalidate_vector_cache()
                self.memory.log_self_teaching(
                    "backup_restore",
                    "Respaldo Completo Restaurado",
                    f"Pesos MHSA, {len(self.memory.episodes)} recuerdos y biblioteca RAG importados.",
                )
                self.save_state()
                return {
                    "ok": True,
                    "message": "Respaldo completo de NeuroZero 3.0 restaurado exitosamente.",
                    "brain_state": self.get_brain_telemetry(),
                }
            except Exception as e:
                return {"ok": False, "message": f"Error al restaurar respaldo: {e}"}
