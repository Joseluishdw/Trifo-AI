"""
Motor Léxico, Morfológico (Stemmer Español desde Cero) y Extractor Autónomo de Conocimiento.
Incluye:
- Tokenización dinámica + raíces morfológicas (stemming en español sin librerías externas)
  para conectar plurales, conjugaciones y derivados automáticamente.
- Tolerancia a errores ortográficos con trigramas de caracteres.
- Transformador gramatical de perspectiva (1ª persona <-> 2ª persona).
- Auto-generador de pares de entrenamiento (Self-Teaching).
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional


STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "sin", "sobre", "y", "e", "o", "u",
    "que", "es", "son", "fue", "era", "ser", "estar", "esta", "este", "esto",
    "estos", "estas", "ese", "esa", "eso", "esos", "esas", "aquel", "aquella",
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "me", "te",
    "se", "nos", "le", "les", "lo", "como", "mas", "pero", "si", "no", "ya",
    "muy", "tan", "tambien", "solo", "entre", "hasta", "desde", "porque", "cuando",
    "donde", "quien", "cual", "todo", "nada", "algo", "asi", "hoy", "ayer",
    "estoy", "estas", "estuve", "estuviste", "tengo", "tienes", "quiero", "quieres",
    "puedo", "puedes", "hago", "haces", "hice", "hiciste", "fui", "fuiste", "voy",
    "vas", "soy", "eres", "sabes", "sabe", "dime", "decir", "hacer", "tener", "ir"
}

QUESTION_STARTERS = {
    "que", "quien", "quienes", "como", "cuando", "donde", "por que", "porque",
    "cual", "cuales", "cuanto", "cuanta", "cuantos", "cuantas", "sabes", "dime",
    "conoces", "explicame", "cuentame"
}


def strip_accents(text: str) -> str:
    """Elimina tildes manteniendo la letra ñ intacta."""
    text = text.replace("ñ", "\x01").replace("Ñ", "\x02")
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    )
    return without_accents.replace("\x01", "ñ").replace("\x02", "Ñ")


def spanish_stem(word: str) -> str:
    w = strip_accents(word.lower().strip())
    if len(w) <= 3:
        return w
    if len(w) == 4:
        if w.endswith("s") and w[-2] in ("a", "e", "o") and w not in {"tres", "pues", "seis"}:
            return w[:-1]
        return w

    long_suffixes = (
        "amientos", "imientos", "aciones", "iciones", "adoras", "adores",
        "amiento", "imiento", "antemente", "iblemente", "mente",
        "acion", "icion", "adora", "ador", "ancias", "encias", "ancia", "encia",
        "idades", "idad", "ivas", "ivos", "iva", "ivo", "osas", "osos", "osa", "oso",
        "andose", "iendose", "arse", "erse", "irse",
        "ando", "iendo", "aron", "ieron", "aban", "ian", "aran", "eran", "iran",
        "aste", "iste", "amos", "imos", "emos", "ar", "er", "ir"
    )
    for suf in long_suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]

    if w.endswith("ces") and len(w) > 5:
        w = w[:-3] + "z"
    elif w.endswith("es") and len(w) > 5 and w[-3] not in ("a", "e", "i", "o", "u"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        w = w[:-1]

    if w[-1] in ("a", "o", "e") and len(w) > 5:
        w = w[:-1]

    return w


class ScratchTokenizer:
    """
    Tokenizador dinámico desde cero con vocabulario incremental,
    raíces morfológicas (stems), IDF dinámico y trigramas de caracteres.
    """

    def __init__(self):
        self.word2idx: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.idx2word: Dict[int, str] = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
        }
        self.stem2idx: Dict[str, int] = {}
        self.word_freq: Dict[str, int] = {}
        self.total_docs: int = 0
        self.doc_freq: Dict[str, int] = {}

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = strip_accents(text)
        text = re.sub(r"[^\w\sñ¿?¡!]", " ", text)
        text = re.sub(r"[¿?¡!]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str, learn_vocab: bool = False) -> List[str]:
        clean = self.normalize(text)
        if not clean:
            return []
        tokens = clean.split(" ")
        if learn_vocab:
            self.total_docs += 1
            seen_in_doc = set()
            for tok in tokens:
                self.register_word(tok)
                self.word_freq[tok] = self.word_freq.get(tok, 0) + 1
                if tok not in seen_in_doc:
                    self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
                    seen_in_doc.add(tok)
        return tokens

    def register_word(self, word: str) -> int:
        if word not in self.word2idx:
            new_id = len(self.word2idx)
            self.word2idx[word] = new_id
            self.idx2word[new_id] = word
            self.word_freq[word] = 0
            self.doc_freq[word] = 0
            stem = spanish_stem(word)
            if stem not in self.stem2idx:
                self.stem2idx[stem] = new_id
        return self.word2idx[word]

    def encode(self, text: str, learn_vocab: bool = False) -> List[int]:
        """
        Convierte texto en IDs enteros. Si una palabra no se ha visto exactamente,
        busca primero por raíz morfológica (stem) y luego por similitud ortográfica!
        """
        tokens = self.tokenize(text, learn_vocab=learn_vocab)
        ids = []
        for t in tokens:
            if t in self.word2idx:
                ids.append(self.word2idx[t])
            else:
                stem = spanish_stem(t)
                if stem in self.stem2idx:
                    ids.append(self.stem2idx[stem])
                else:
                    match = self.find_closest_known_word(t)
                    if match is not None:
                        ids.append(self.word2idx[match])
                    else:
                        ids.append(self.word2idx["<UNK>"])
        return ids

    def get_idf_weights(self, tokens: List[str]) -> List[float]:
        """Calcula importancia léxica IDF para cada token de la secuencia."""
        import math
        weights = []
        n_docs = max(1, self.total_docs)
        for t in tokens:
            if t in STOPWORDS_ES:
                weights.append(0.35)
            else:
                df = self.doc_freq.get(t, 0)
                idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
                weights.append(min(2.5, idf))
        return weights

    def char_ngrams(self, word: str, n: int = 3) -> set:
        padded = f"#{word}#"
        if len(padded) < n:
            return {padded}
        return {padded[i:i + n] for i in range(len(padded) - n + 1)}

    def word_similarity(self, w1: str, w2: str) -> float:
        if w1 == w2:
            return 1.0
        if spanish_stem(w1) == spanish_stem(w2) and len(w1) >= 4 and len(w2) >= 4:
            return 0.88
        if abs(len(w1) - len(w2)) > 3 or len(w1) < 3 or len(w2) < 3:
            return 0.0
        g1 = self.char_ngrams(w1, 3)
        g2 = self.char_ngrams(w2, 3)
        inter = len(g1 & g2)
        union = len(g1 | g2)
        return inter / union if union > 0 else 0.0

    def find_closest_known_word(self, unknown_word: str, threshold: float = 0.62) -> Optional[str]:
        if len(unknown_word) < 4:
            return None
        best_word = None
        best_score = threshold
        for known in self.word2idx:
            if known.startswith("<"):
                continue
            score = self.word_similarity(unknown_word, known)
            if score > best_score:
                best_score = score
                best_word = known
        return best_word

    def extract_keywords(self, text: str) -> List[str]:
        tokens = self.tokenize(text, learn_vocab=False)
        keywords = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
        if not keywords:
            keywords = [t for t in tokens if len(t) > 1]
        return keywords

    def to_dict(self) -> dict:
        return {
            "word2idx": self.word2idx,
            "word_freq": self.word_freq,
            "total_docs": self.total_docs,
            "doc_freq": self.doc_freq,
        }

    def from_dict(self, data: dict):
        self.word2idx = data.get("word2idx", self.word2idx)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.word_freq = data.get("word_freq", {})
        self.total_docs = int(data.get("total_docs", 0))
        self.doc_freq = data.get("doc_freq", {})
        self.stem2idx = {}
        for w, idx in self.word2idx.items():
            if not w.startswith("<"):
                st = spanish_stem(w)
                if st not in self.stem2idx:
                    self.stem2idx[st] = int(idx)


class NaturalChatParser:
    """
    Analizador conversacional autónomo:
    - Detecta preguntas, reglas lógicas ("Si X entonces Y"), correcciones y afirmaciones.
    - Transforma perspectiva gramatical (1ª persona <-> 2ª persona).
    - Auto-genera pares de entrenamiento internos (Self-Teaching).
    """

    PERSPECTIVE_MAP = {
        "yo": "tú",
        "mi": "tu",
        "mis": "tus",
        "mio": "tuyo",
        "mia": "tuya",
        "me": "te",
        "conmigo": "contigo",
        "estoy": "estás",
        "soy": "eres",
        "tengo": "tienes",
        "quiero": "quieres",
        "puedo": "puedes",
        "hago": "haces",
        "voy": "vas",
        "fui": "fuiste",
        "estuve": "estuviste",
        "hice": "hiciste",
        "comi": "comiste",
        "vi": "viste",
        "trabajo": "trabajas",
        "estudio": "estudias",
        "vivo": "vives",
        "siento": "sientes",
        "pienso": "piensas",
        "creo": "crees",
        "necesito": "necesitas",
        "busco": "buscas",
        "prefiero": "prefieres",
        "amo": "amas",
        "odio": "odias",
        "juego": "juegas",
        "programo": "programas",
    }

    @classmethod
    def shift_user_to_second_person(cls, text: str) -> str:
        words = re.sub(r"[.!?¡¿]+$", "", text.strip()).split()
        shifted = []
        for w in words:
            clean_w = strip_accents(w.lower().strip(",.;:"))
            punct = w[len(w.rstrip(",.;:")):]
            if clean_w in cls.PERSPECTIVE_MAP:
                shifted.append(cls.PERSPECTIVE_MAP[clean_w] + punct)
            else:
                shifted.append(w)
        res = " ".join(shifted)
        res = re.sub(r"\ba\s+mi\s+te\b", "a ti te", res, flags=re.IGNORECASE)
        return res

    @staticmethod
    def is_question(text: str) -> bool:
        raw = text.strip()
        if "?" in raw or "¿" in raw:
            return True
        norm = strip_accents(raw.lower())
        first_words = norm.split()[:2]
        if not first_words:
            return False
        if first_words[0] in QUESTION_STARTERS:
            return True
        if len(first_words) >= 2 and f"{first_words[0]} {first_words[1]}" in QUESTION_STARTERS:
            return True
        return False

    @staticmethod
    def detect_explicit_teaching(text: str) -> Optional[Tuple[str, str]]:
        raw = text.strip()
        patterns = [
            r"(?i)^(?:cuando\s+(?:te\s+)?diga|si\s+(?:te\s+)?digo|si\s+pregunto)\s+[\"']?(.+?)[\"']?\s+(?:responde|di|dime|contesta)\s*:?\s*[\"']?(.+?)[\"']?$",
            r"(?i)^(?:aprende|entrena)\s*:?\s*(.+?)\s*(?:->|=>|=)\s*(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, raw)
            if m:
                trigger = m.group(1).strip()
                response = m.group(2).strip()
                if trigger and response:
                    return (trigger, response)
        return None

    @staticmethod
    def detect_logical_rule(text: str) -> Optional[Tuple[str, str]]:
        """
        Detecta reglas condicionales lógicas:
        - "Si X entonces Y"
        - "Todo X es Y" / "Todos los X son Y"
        """
        raw = re.sub(r"[.!?¡¿]+$", "", text.strip())
        m_cond = re.match(r"(?i)^si\s+(.+?)\s+(?:entonces|implica\s+que|significa\s+que|,)\s*(.+)$", raw)
        if m_cond:
            cond = m_cond.group(1).strip()
            cons = m_cond.group(2).strip()
            # Evitar confundir con "si te digo X responde Y"
            if not re.match(r"(?i)^(?:te\s+)?digo\b", cond):
                return (cond, cons)
        return None

    @staticmethod
    def detect_chat_correction(text: str) -> Optional[str]:
        raw = text.strip()
        patterns = [
            r"(?i)^(?:no\s*,?\s*)?(?:eso\s+esta\s+mal\s*,?\s*)?(?:deberias\s+(?:decir|responder)|la\s+respuesta\s+(?:correcta\s+)?es|mejor\s+responde|mejor\s+di|corrige(?:\s+eso)?|di\s+mejor)\s*:?\s*[\"']?(.+?)[\"']?$",
            r"(?i)^no\s*,\s*(?:responde|di|contesta)\s*:?\s*[\"']?(.+?)[\"']?$",
        ]
        for pat in patterns:
            m = re.match(pat, raw)
            if m:
                correction = m.group(1).strip()
                if len(correction) >= 2:
                    return correction
        return None

    @classmethod
    def extract_fact(cls, text: str) -> Optional[Dict[str, str]]:
        raw = text.strip()
        clean = re.sub(r"[.!?¡¿]+$", "", raw).strip()

        if cls.is_question(raw):
            return None

        # Ignorar comandos de herramientas/investigación para no guardarlos como hechos del mundo
        norm_start = strip_accents(clean.lower())
        if re.match(r"^(investiga|busca|calcula|cuanto|resuelve|agrega\s+tarea|anota|recuerdame|guarda\s+nota|crea\s+archivo)\b", norm_start):
            return None

        # 1. Hechos sobre el usuario
        user_patterns = [
            (r"(?i)^(?:hola\s*,?\s*)?(?:me\s+llamo|mi\s+nombre\s+es)\s+(.+)$", "usuario", "nombre"),
            (r"(?i)^(?:yo\s+)?tengo\s+(\d+\s*a[ñn]os)$", "usuario", "edad"),
            (r"(?i)^(?:a\s+mi\s+)?me\s+gusta(?:n)?\s+(.+)$", "usuario", "gusto"),
            (r"(?i)^(?:a\s+mi\s+)?me\s+encanta(?:n)?\s+(.+)$", "usuario", "gusto"),
            (r"(?i)^mi\s+([a-záéíóúüñ\s]{2,20}?)\s+favorit[oa]\s+es\s+(.+)$", "usuario_pref", "favorito"),
            (r"(?i)^vivo\s+en\s+(.+)$", "usuario", "ubicacion"),
            (r"(?i)^(?:yo\s+)?trabajo\s+(?:en|como|de)\s+(.+)$", "usuario", "trabajo"),
            (r"(?i)^(?:yo\s+)?estudio\s+(.+)$", "usuario", "estudio"),
            (r"(?i)^(?:hoy\s+|ahora\s+)?estoy\s+(.+)$", "usuario", "estado_actual"),
            (r"(?i)^(?:yo\s+)?quiero\s+(.+)$", "usuario", "deseo"),
            (r"(?i)^soy\s+(?:un|una)?\s*([a-záéíóúüñ\s]{3,40})$", "usuario", "identidad"),
        ]
        for pat, subj_type, rel in user_patterns:
            m = re.match(pat, clean)
            if m:
                if subj_type == "usuario_pref":
                    pref_topic = strip_accents(m.group(1).strip().lower())
                    val = m.group(2).strip()
                    return {
                        "subject": f"tu {pref_topic} favorito",
                        "relation": "es",
                        "object": val,
                        "category": "user",
                    }
                val = m.group(1).strip()
                return {
                    "subject": "usuario",
                    "relation": rel,
                    "object": val,
                    "category": "user",
                }

        # 2. Hechos sobre la propia IA
        ai_patterns = [
            (r"(?i)^(?:tu\s+nombre\s+es|te\s+llamas|ahora\s+te\s+llamas)\s+(.+)$", "ia", "nombre"),
            (r"(?i)^(?:tu\s+)?eres\s+(?:una?\s+)?(.+)$", "ia", "identidad"),
        ]
        for pat, subj, rel in ai_patterns:
            m = re.match(pat, clean)
            if m:
                val = m.group(1).strip()
                if len(val) > 1:
                    return {
                        "subject": "ia",
                        "relation": rel,
                        "object": val,
                        "category": "ai",
                    }

        neg_univ_m = re.match(
            r"(?i)^(?:ningun[oa]?)\s+([a-záéíóúñ0-9\s]{1,35}?)\s+(es\s+un|es\s+una|es|son|tiene|tienen|puede|pueden|necesita|necesitan)\s+(.+)$",
            strip_accents(clean),
        )
        if neg_univ_m:
            return {
                "subject": neg_univ_m.group(1).strip(),
                "relation": f"no {neg_univ_m.group(2).strip().lower()}",
                "object": neg_univ_m.group(3).strip(),
                "category": "world",
            }

        univ_m = re.match(
            r"(?i)^(?:tod[oa]s?\s+(?:los\s+|las\s+)?|cada\s+)([a-záéíóúüñ0-9\s]{1,35}?)\s+(es\s+un|es\s+una|es|son|tiene|tienen|puede|pueden|necesita|necesitan|no\s+puede(?:n)?|no\s+necesita(?:n)?|no\s+es|no\s+son)\s+(.+)$",
            clean,
        )
        if univ_m:
            return {
                "subject": univ_m.group(1).strip(),
                "relation": univ_m.group(2).strip().lower(),
                "object": univ_m.group(3).strip(),
                "category": "world",
            }

        rel_expr = (
            r"no\s+es\s+un|no\s+es\s+una|no\s+es|no\s+son|no\s+tiene(?:n)?|no\s+puede(?:n)?|no\s+necesita(?:n)?|"
            r"es\s+un|es\s+una|es\s+el|es\s+la|es|son|significa|tiene|tienen|puede|pueden|"
            r"sirve\s+para|sirven\s+para|funciona\s+para|funciona\s+con|ayuda\s+a|queda\s+en|esta\s+en|"
            r"vive\s+en|viven\s+en|come|comen|usa|usan|hace|hacen|fue|produce|causa|necesita|necesitan|pertenece\s+a"
        )
        world_patterns = [
            rf"(?i)^(?:recuerda\s+que|aprende\s+que|sabias\s+que|ten\s+en\s+cuenta\s+que|fijate\s+que)\s+(?:(?:el|la|los|las|un|una)\s+)?(.+?)\s+({rel_expr})\s+(.+)$",
            rf"(?i)^(?:(?:el|la|los|las|un|una)\s+)?([a-záéíóúüñ0-9][a-záéíóúüñ0-9\s]{{0,34}}?)\s+({rel_expr})\s+(.+)$",
        ]
        for pat in world_patterns:
            m = re.match(pat, clean)
            if m:
                subj = m.group(1).strip()
                rel = m.group(2).strip().lower()
                obj = m.group(3).strip()
                subj_norm = strip_accents(subj.lower())
                invalid_subjects = {
                    "eso", "esto", "aquello", "que", "como", "cual", "quien",
                    "todo", "nada", "algo", "hola", "bien", "mal", "si", "no",
                    "tu", "yo", "el", "ella", "mi", "un", "una", "la", "los", "las",
                    "me", "te", "se", "lo", "le"
                }
                if subj_norm not in invalid_subjects and len(subj) >= 1 and len(obj) >= 1:
                    return {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "category": "world",
                    }
        return None

    @classmethod
    def generate_self_teaching_pairs(
        cls, user_text: str, keywords: List[str], extracted_fact: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, str, str]]:
        clean = re.sub(r"[.!?¡¿]+$", "", user_text.strip()).strip()
        if len(clean) < 4 or cls.is_question(user_text):
            return []

        pairs: List[Tuple[str, str, str]] = []
        norm = strip_accents(clean.lower())

        if extracted_fact:
            subj = extracted_fact["subject"]
            rel = extracted_fact["relation"]
            obj = extracted_fact["object"]
            cat = extracted_fact["category"]

            if cat == "user":
                if rel == "nombre":
                    pairs.append(("como me llamo", f"Te llamas {obj}.", f"Auto-identidad: {obj}"))
                elif rel == "gusto":
                    pairs.append(("que me gusta", f"Me contaste que te gusta {obj}.", f"Auto-gusto: {obj}"))
                elif rel == "estado_actual":
                    shifted = cls.shift_user_to_second_person(f"estoy {obj}")
                    pairs.append(("que estoy haciendo", f"Me dijiste que {shifted}.", f"Auto-actividad: {obj}"))
            elif cat == "world":
                subj_clean = re.sub(r"^(el|la|los|las|un|una)\s+", "", subj, flags=re.IGNORECASE).strip()
                ans = f"{subj.capitalize()} {rel} {obj}."
                pairs.append((f"que es {subj_clean}", ans, f"Auto-definición: {subj_clean}"))
                pairs.append((f"hablame de {subj_clean}", f"Sé que {subj.lower()} {rel} {obj}.", f"Auto-síntesis: {subj_clean}"))

        first_person_markers = {
            "yo", "mi", "mis", "me", "estoy", "estuve", "fui", "hice",
            "tengo", "quiero", "programo", "trabajo", "estudio", "creo", "pienso"
        }
        tokens_norm = norm.split()
        is_first_person = any(t in first_person_markers for t in tokens_norm)

        if is_first_person:
            shifted_sentence = cls.shift_user_to_second_person(clean)
            for kw in keywords[:3]:
                if len(kw) >= 4:
                    pairs.append(
                        (
                            f"que sabes de {kw}",
                            f"Recuerdo que me comentaste que {shifted_sentence}.",
                            f"Auto-deducción personal sobre «{kw}»",
                        )
                    )
        else:
            for kw in keywords[:3]:
                if len(kw) >= 4:
                    pairs.append(
                        (
                            f"que sabes de {kw}",
                            f"Sé que {clean.lower()}.",
                            f"Auto-enseñanza sobre «{kw}»",
                        )
                    )

        seen = set()
        unique_pairs = []
        for trig, resp, desc in pairs:
            key = (strip_accents(trig.lower()), strip_accents(resp.lower()))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((trig, resp, desc))

        return unique_pairs[:4]
