"""
Sistema de Memoria Híbrida y Auto-Aprendizaje desde Cero:
1. Memoria Episódica Asociativa: Experiencias (estímulo -> respuesta) tanto enseñadas
   como auto-generadas por la propia IA al escucharte.
2. Grafo de Conocimiento Semántico y Red de Conceptos Asociativos: Conecta cada palabra/concepto
   importante que mencionas con tus ideas y con otros conceptos relacionados.
3. Generador Neuronal-Markov (Bigramas + Trigramas): Permite a la IA construir frases
   nuevas usando el vocabulario y estilo que aprende de ti.
4. Bitácora de Auto-Enseñanza (Self-Teaching Log): Registra en vivo cada deducción,
   auto-pregunta y conexión que la IA aprende por sí sola.
"""

import time
import random
from typing import List, Dict, Optional, Tuple
from .tokenizer import strip_accents, STOPWORDS_ES, NaturalChatParser


class EpisodicMemoryItem:
    def __init__(
        self,
        item_id: int,
        stimulus: str,
        response: str,
        context: str = "",
        reward: float = 1.0,
        uses: int = 0,
        source: str = "chat",
        created_at: Optional[float] = None,
    ):
        self.id = item_id
        self.stimulus = stimulus
        self.response = response
        self.context = context
        self.reward = reward
        self.uses = uses
        self.source = source  # "seed", "auto_self_taught", "conversational_mirror", "corrected"
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stimulus": self.stimulus,
            "response": self.response,
            "context": self.context,
            "reward": round(float(self.reward), 3),
            "uses": int(self.uses),
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicMemoryItem":
        return cls(
            item_id=int(d["id"]),
            stimulus=d["stimulus"],
            response=d["response"],
            context=d.get("context", ""),
            reward=float(d.get("reward", 1.0)),
            uses=int(d.get("uses", 0)),
            source=d.get("source", "chat"),
            created_at=float(d.get("created_at", time.time())),
        )


class HybridMemorySystem:
    """
    Gestiona:
    - Memoria Episódica
    - Grafo de Conocimiento Semántico (Sujeto-Relación-Objeto)
    - Grafo Asociativo de Conceptos Libres (para aprender de cualquier frase del usuario)
    - Generador Secuencial Bigrama/Trigrama (Markov-Neuronal)
    - Bitácora de Auto-Enseñanza en vivo
    """

    def __init__(self):
        self.episodes: List[EpisodicMemoryItem] = []
        self.next_id: int = 1

        self.facts: List[Dict[str, object]] = []
        self.next_fact_id: int = 1

        # Red asociativa de conceptos: {concepto: {concepto_vecino: peso}}
        self.concept_links: Dict[str, Dict[str, int]] = {}
        # Ideas/frases aprendidas asociadas a cada concepto: {concepto: [frase1, frase2]}
        self.concept_memories: Dict[str, List[str]] = {}

        # Transiciones de palabras para generación propia
        self.markov_bigrams: Dict[str, Dict[str, int]] = {}
        self.markov_trigrams: Dict[str, Dict[str, int]] = {}

        # Bitácora de Auto-Enseñanza (lo que la IA va deduciendo y aprendiendo sola)
        self.self_teaching_log: List[Dict[str, object]] = []

    def log_self_teaching(self, kind: str, summary: str, detail: str, loss: Optional[float] = None):
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "kind": kind,
            "summary": summary,
            "detail": detail,
            "loss": round(loss, 5) if loss is not None else None,
        }
        self.self_teaching_log.append(entry)
        if len(self.self_teaching_log) > 80:
            self.self_teaching_log = self.self_teaching_log[-80:]

    def add_episode(
        self,
        stimulus: str,
        response: str,
        context: str = "",
        reward: float = 1.0,
        source: str = "auto_self_taught",
    ) -> EpisodicMemoryItem:
        stim_norm = strip_accents(stimulus.lower().strip())
        resp_norm = strip_accents(response.lower().strip())

        self.learn_markov_sequence(response)

        for ep in self.episodes:
            if (
                strip_accents(ep.stimulus.lower().strip()) == stim_norm
                and strip_accents(ep.response.lower().strip()) == resp_norm
            ):
                ep.reward = min(3.0, ep.reward + 0.25)
                return ep

        item = EpisodicMemoryItem(
            item_id=self.next_id,
            stimulus=stimulus.strip(),
            response=response.strip(),
            context=context.strip(),
            reward=reward,
            uses=0,
            source=source,
        )
        self.next_id += 1
        self.episodes.append(item)
        return item

    def absorb_user_statement_into_concepts(self, user_text: str, keywords: List[str]):
        """
        Absorbe libremente cualquier frase que el usuario dice:
        1. Conecta entre sí todos los conceptos clave de la frase en `concept_links`.
        2. Guarda la idea (con perspectiva transformada si hablaba de sí mismo) en `concept_memories`.
        3. Alimenta el motor generativo de secuencias (Markov bigrama + trigrama).
        """
        self.learn_markov_sequence(user_text)
        if not keywords:
            return

        # Conectar conceptos co-ocurrentes en el grafo asociativo
        for i in range(len(keywords)):
            c1 = keywords[i]
            if c1 not in self.concept_links:
                self.concept_links[c1] = {}
            for j in range(len(keywords)):
                if i == j:
                    continue
                c2 = keywords[j]
                self.concept_links[c1][c2] = self.concept_links[c1].get(c2, 0) + 1

        # Si no es pregunta, guardar la idea asociada a cada concepto clave
        if not NaturalChatParser.is_question(user_text) and len(user_text.strip()) >= 5:
            shifted = NaturalChatParser.shift_user_to_second_person(user_text.strip())
            for kw in keywords[:4]:
                if len(kw) < 3:
                    continue
                if kw not in self.concept_memories:
                    self.concept_memories[kw] = []
                if shifted not in self.concept_memories[kw]:
                    self.concept_memories[kw].append(shifted)
                    if len(self.concept_memories[kw]) > 8:
                        self.concept_memories[kw] = self.concept_memories[kw][-8:]

    def get_related_concepts(self, concept: str, top_k: int = 3) -> List[str]:
        c_norm = strip_accents(concept.lower().strip())
        links = self.concept_links.get(c_norm, {})
        if not links:
            return []
        sorted_links = sorted(links.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_links[:top_k]]

    def add_or_update_fact(
        self, subject: str, relation: str, obj: str, category: str = "world"
    ) -> Dict[str, object]:
        subj_norm = strip_accents(subject.lower().strip())
        rel_norm = strip_accents(relation.lower().strip())

        for f in self.facts:
            f_subj = strip_accents(str(f["subject"]).lower().strip())
            f_rel = strip_accents(str(f["relation"]).lower().strip())
            if f_subj == subj_norm and f_rel == rel_norm and rel_norm in {"nombre", "edad", "ubicacion", "estado_actual", "es"}:
                f["object"] = obj.strip()
                f["updated_at"] = time.time()
                return f

        new_fact = {
            "id": self.next_fact_id,
            "subject": subject.strip(),
            "relation": relation.strip(),
            "object": obj.strip(),
            "category": category,
            "updated_at": time.time(),
        }
        self.next_fact_id += 1
        self.facts.append(new_fact)
        self.learn_markov_sequence(f"{subject} {relation} {obj}")
        return new_fact

    def query_facts_and_concepts(self, user_text: str, keywords: List[str]) -> Optional[Tuple[str, Dict[str, object]]]:
        """
        Consulta tanto el Grafo de Conocimiento Estructurado como la Memoria de Conceptos Libres
        para responder preguntas sobre lo que el usuario le ha platicado.
        """
        norm = strip_accents(user_text.lower().strip())

        # 1. Preguntas sobre el usuario
        if any(p in norm for p in ["como me llamo", "cual es mi nombre", "sabes mi nombre", "quien soy"]):
            for f in reversed(self.facts):
                if f["category"] == "user" and f["relation"] == "nombre":
                    return (f"¡Te llamas {f['object']}! Lo aprendí cuando me lo dijiste.", f)
            for f in reversed(self.facts):
                if f["category"] == "user" and f["relation"] == "identidad":
                    return (f"Sé que eres {f['object']}.", f)

        if any(p in norm for p in ["que me gusta", "mis gustos", "que me encanta", "sabes que me gusta"]):
            gustos = [str(f["object"]) for f in self.facts if f["category"] == "user" and f["relation"] == "gusto"]
            if gustos:
                lista = ", ".join(gustos[-3:])
                return (f"Por lo que me has platicado, te gusta: {lista}.", self.facts[-1])

        if any(p in norm for p in ["que estoy haciendo", "que hago", "que ando haciendo", "que hice"]):
            for f in reversed(self.facts):
                if f["category"] == "user" and f["relation"] == "estado_actual":
                    return (f"Me contaste que estás {f['object']}.", f)

        if any(p in norm for p in ["donde vivo", "en donde vivo", "cual es mi ciudad"]):
            for f in reversed(self.facts):
                if f["category"] == "user" and f["relation"] == "ubicacion":
                    return (f"Me dijiste que vives en {f['object']}.", f)

        if any(p in norm for p in ["que sabes de mi", "que recuerdas de mi", "conoces algo de mi"]):
            user_facts = [f for f in self.facts if f["category"] == "user"]
            if user_facts:
                parts = []
                for uf in user_facts[-4:]:
                    if uf["relation"] == "nombre":
                        parts.append(f"te llamas {uf['object']}")
                    elif uf["relation"] == "gusto":
                        parts.append(f"te gusta {uf['object']}")
                    elif uf["relation"] == "estado_actual":
                        parts.append(f"estás {uf['object']}")
                    else:
                        parts.append(f"tu {uf['relation']} es {uf['object']}")
                return (f"De lo que hemos hablado, sé que: {', '.join(parts)}.", user_facts[-1])

        # 2. Preguntas sobre la propia IA
        if any(p in norm for p in ["como te llamas", "cual es tu nombre", "quien eres", "que eres"]):
            ai_name = None
            ai_id = None
            for f in reversed(self.facts):
                if f["category"] == "ai" and f["relation"] == "nombre" and not ai_name:
                    ai_name = f["object"]
                if f["category"] == "ai" and f["relation"] == "identidad" and not ai_id:
                    ai_id = f["object"]
            if ai_name and ai_id:
                return (f"Me llamo {ai_name} y soy {ai_id}. Me auto-entreno con todo lo que hablamos.", f)
            elif ai_name:
                return (f"Me llamo {ai_name}.", f)

        # 3. Preguntas sobre cualquier hecho o concepto aprendido del usuario
        if NaturalChatParser.is_question(user_text):
            for f in reversed(self.facts):
                if f["category"] in {"world", "user"}:
                    subj_norm = strip_accents(str(f["subject"]).lower().strip())
                    subj_clean = subj_norm
                    for art in ("el ", "la ", "los ", "las ", "un ", "una ", "tu ", "mi "):
                        if subj_clean.startswith(art):
                            subj_clean = subj_clean[len(art):].strip()
                    if len(subj_clean) >= 2 and subj_clean in norm:
                        subj_display = str(f["subject"]).capitalize()
                        return (f"{subj_display} {f['relation']} {f['object']}.", f)

            # Si no está como triple exacto, buscar en los conceptos libres (`concept_memories`)
            for kw in keywords:
                if kw in self.concept_memories and self.concept_memories[kw]:
                    idea = self.concept_memories[kw][-1]
                    related = self.get_related_concepts(kw, top_k=2)
                    rel_str = f" (también lo relaciono con {', '.join(related)})" if related else ""
                    return (
                        f"Sobre «{kw}», recuerdo lo que me dijiste: {idea}.{rel_str}",
                        {"subject": kw, "relation": "asociado_a", "object": idea, "category": "concept"},
                    )

        return None

    def learn_markov_sequence(self, text: str):
        """Entrena las tablas de transición de Bigramas y Trigramas con cada frase."""
        words = text.strip().split()
        if len(words) < 2:
            return
        tokens = ["<BOS>"] + words + ["<EOS>"]
        for i in range(len(tokens) - 1):
            w1 = strip_accents(tokens[i].lower().strip(",.;:!?¡¿"))
            w2 = tokens[i + 1]
            if not w1:
                continue
            if w1 not in self.markov_bigrams:
                self.markov_bigrams[w1] = {}
            self.markov_bigrams[w1][w2] = self.markov_bigrams[w1].get(w2, 0) + 1

            if i < len(tokens) - 2:
                w2_norm = strip_accents(tokens[i + 1].lower().strip(",.;:!?¡¿"))
                w3 = tokens[i + 2]
                key_tri = f"{w1}|{w2_norm}"
                if key_tri not in self.markov_trigrams:
                    self.markov_trigrams[key_tri] = {}
                self.markov_trigrams[key_tri][w3] = self.markov_trigrams[key_tri].get(w3, 0) + 1

    def generate_sentence_from_seed(self, seed_words: List[str], max_len: int = 12) -> Optional[str]:
        """
        Genera una frase propia usando las cadenas de Markov (Trigramas + Bigramas)
        iniciadas desde un concepto semilla que el usuario mencionó.
        """
        valid_seeds = [
            strip_accents(w.lower()) for w in seed_words if strip_accents(w.lower()) in self.markov_bigrams
        ]
        if not valid_seeds:
            return None

        start = random.choice(valid_seeds)
        chain = [start]
        curr = start

        for _ in range(max_len):
            next_word = None
            if len(chain) >= 2:
                prev_norm = strip_accents(chain[-2].lower().strip(",.;:!?¡¿"))
                curr_norm = strip_accents(chain[-1].lower().strip(",.;:!?¡¿"))
                tri_candidates = self.markov_trigrams.get(f"{prev_norm}|{curr_norm}", {})
                if tri_candidates:
                    words, weights = zip(*tri_candidates.items())
                    next_word = random.choices(words, weights=weights, k=1)[0]

            if next_word is None:
                bi_candidates = self.markov_bigrams.get(curr, {})
                if not bi_candidates:
                    break
                words, weights = zip(*bi_candidates.items())
                next_word = random.choices(words, weights=weights, k=1)[0]

            if next_word == "<EOS>":
                break
            chain.append(next_word)
            curr = strip_accents(next_word.lower().strip(",.;:!?¡¿"))

        if len(chain) >= 4:
            sentence = " ".join(chain)
            sentence = NaturalChatParser.shift_user_to_second_person(sentence)
            return sentence[0].upper() + sentence[1:]
        return None

    def get_user_name(self) -> Optional[str]:
        for f in reversed(self.facts):
            if f["category"] == "user" and f["relation"] == "nombre":
                return str(f["object"])
        return None

    def to_dict(self) -> dict:
        return {
            "next_id": self.next_id,
            "next_fact_id": self.next_fact_id,
            "episodes": [ep.to_dict() for ep in self.episodes],
            "facts": self.facts,
            "concept_links": self.concept_links,
            "concept_memories": self.concept_memories,
            "markov_bigrams": self.markov_bigrams,
            "markov_trigrams": self.markov_trigrams,
            "self_teaching_log": self.self_teaching_log,
        }

    def from_dict(self, data: dict):
        self.next_id = int(data.get("next_id", 1))
        self.next_fact_id = int(data.get("next_fact_id", 1))
        self.episodes = [EpisodicMemoryItem.from_dict(d) for d in data.get("episodes", [])]
        self.facts = data.get("facts", [])
        self.concept_links = data.get("concept_links", {})
        self.concept_memories = data.get("concept_memories", {})
        self.markov_bigrams = data.get("markov_bigrams", {})
        self.markov_trigrams = data.get("markov_trigrams", {})
        self.self_teaching_log = data.get("self_teaching_log", [])
