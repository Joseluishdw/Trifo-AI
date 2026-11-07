"""
NeuroZero 3.0 — Motor RAG Vectorial con Caché Persistente + Lector Multi-Formato + Wikipedia.
Incluye:
- Caché persistente en SQLite + matriz vectorizada en RAM de IDs tokenizados y embeddings
  de cada fragmento, con actualización automática cuando el Transformer entrena nuevas épocas.
- Sincronización Thread-Safe mediante threading.RLock y SQLite WAL.
- Lector desde cero de .txt, .md, .csv (con resumen estadístico NumPy), .py, .json, .html, .docx y .pdf.
- Auto-investigador autónomo en Wikipedia en Español.
"""

import io
import os
import re
import csv
import json
import zlib
import time
import sqlite3
import zipfile
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
import numpy as np

from .tokenizer import strip_accents, spanish_stem, STOPWORDS_ES


class DocumentParserFromScratch:
    """Extrae texto limpio y metadatos estadísticos usando únicamente librería estándar + NumPy."""

    @staticmethod
    def parse_bytes(filename: str, raw_bytes: bytes) -> str:
        ext = os.path.splitext(filename.lower())[1]

        if ext == ".docx":
            return DocumentParserFromScratch._parse_docx(raw_bytes)
        if ext == ".pdf":
            return DocumentParserFromScratch._parse_pdf_basic(raw_bytes)
        if ext == ".csv":
            return DocumentParserFromScratch._parse_csv_with_stats(filename, raw_bytes)
        if ext in {".html", ".htm"}:
            text = DocumentParserFromScratch._decode_text(raw_bytes)
            text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        return DocumentParserFromScratch._decode_text(raw_bytes)

    @staticmethod
    def analyze_csv_statistics(raw_bytes: bytes) -> Optional[Dict[str, object]]:
        """Calcula media, mediana, desviación estándar, mínimo y máximo por columna numérica con NumPy."""
        text = DocumentParserFromScratch._decode_text(raw_bytes)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None

        sample = "\n".join(lines[:10])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter
        except Exception:
            delim = ","

        reader = list(csv.reader(lines, delimiter=delim))
        if len(reader) < 2:
            return None

        headers = [h.strip() or f"col_{i+1}" for i, h in enumerate(reader[0])]
        data_rows = reader[1:]
        num_cols = len(headers)

        col_stats: Dict[str, Dict[str, float]] = {}
        for c_idx in range(num_cols):
            vals = []
            for row in data_rows:
                if c_idx < len(row):
                    cell = row[c_idx].strip().replace(",", "")
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        continue
            if len(vals) >= 2:
                arr = np.array(vals, dtype=np.float64)
                col_stats[headers[c_idx]] = {
                    "count": int(arr.size),
                    "mean": round(float(np.mean(arr)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                    "min": round(float(np.min(arr)), 4),
                    "max": round(float(np.max(arr)), 4),
                }

        return {
            "rows_count": len(data_rows),
            "columns": headers,
            "numeric_stats": col_stats,
        }

    @staticmethod
    def _parse_csv_with_stats(filename: str, raw_bytes: bytes) -> str:
        raw_text = DocumentParserFromScratch._decode_text(raw_bytes)
        stats = DocumentParserFromScratch.analyze_csv_statistics(raw_bytes)
        if not stats:
            return raw_text

        summary_lines = [
            f"Análisis estadístico del archivo CSV {filename}: {stats['rows_count']} filas y {len(stats['columns'])} columnas ({', '.join(stats['columns'])})."
        ]
        for col_name, st in stats["numeric_stats"].items():
            summary_lines.append(
                f"Columna {col_name} (n={st['count']}): media={st['mean']}, mediana={st['median']}, "
                f"desviación_estándar={st['std']}, mínimo={st['min']}, máximo={st['max']}."
            )

        preview_rows = "\n".join(raw_text.splitlines()[:80])
        return "\n".join(summary_lines) + "\n\n Muestra de filas del CSV:\n" + preview_rows

    @staticmethod
    def _decode_text(raw_bytes: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return raw_bytes.decode(enc)
            except Exception:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def _parse_docx(raw_bytes: bytes) -> str:
        paragraphs = []
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            root = ET.fromstring(zf.read("word/document.xml"))
            for elem in root.iter():
                if elem.tag.endswith("}p"):
                    texts = [t.text for t in elem.iter() if t.tag.endswith("}t") and t.text]
                    if texts:
                        paragraphs.append("".join(texts))
        return "\n".join(paragraphs)

    @staticmethod
    def _parse_pdf_basic(raw_bytes: bytes) -> str:
        extracted_parts = []
        streams = re.findall(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw_bytes, re.DOTALL)
        for s in streams:
            data = s
            for candidate in (s, s.strip()):
                try:
                    data = zlib.decompress(candidate)
                    break
                except Exception:
                    continue
            for tc in re.findall(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*Tj", data):
                extracted_parts.append(tc.decode("latin-1", errors="ignore"))
            for ac in re.findall(rb"\[(.*?)\]\s*TJ", data, re.DOTALL):
                sub_strs = re.findall(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)", ac)
                if sub_strs:
                    extracted_parts.append("".join(x.decode("latin-1", errors="ignore") for x in sub_strs))
        return re.sub(r"\s+", " ", " ".join(extracted_parts)).strip()


class WikipediaAutoResearcher:
    USER_AGENT = "NeuroZeroFromScratch/3.0 (Educational Neuro-Symbolic AI; Python urllib)"

    @classmethod
    def search_and_fetch(cls, query: str) -> Optional[Dict[str, str]]:
        clean_q = re.sub(
            r"(?i)^(?:investiga(?:\s+sobre|\s+que\s+es|\s+quien\s+es)?|busca(?:\s+en\s+wikipedia)?(?:\s+sobre)?|que\s+es|quien\s+es|quienes\s+son|hablame\s+de|explicame\s+que\s+es)\s+",
            "",
            query.strip(),
        )
        clean_q = re.sub(r"[?¿!¡.]+$", "", clean_q).strip()
        if len(clean_q) < 2:
            return None

        try:
            search_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_q,
                    "srlimit": 1,
                    "format": "json",
                }
            )
            req = urllib.request.Request(
                f"https://es.wikipedia.org/w/api.php?{search_params}",
                headers={"User-Agent": cls.USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=7) as resp:
                search_data = json.loads(resp.read().decode("utf-8"))

            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return None

            best_title = hits[0]["title"]
            extract_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exchars": 4500,
                    "titles": best_title,
                    "redirects": 1,
                    "format": "json",
                }
            )
            req2 = urllib.request.Request(
                f"https://es.wikipedia.org/w/api.php?{extract_params}",
                headers={"User-Agent": cls.USER_AGENT},
            )
            with urllib.request.urlopen(req2, timeout=8) as resp2:
                ext_data = json.loads(resp2.read().decode("utf-8"))

            pages = ext_data.get("query", {}).get("pages", {})
            for _, page_info in pages.items():
                extract = page_info.get("extract", "").strip()
                if extract:
                    extract = re.sub(r"==+\s*[^=]+\s*==+", "\n", extract)
                    extract = re.sub(r"\n{2,}", "\n", extract).strip()
                    return {
                        "title": page_info.get("title", best_title),
                        "content": extract,
                        "url": f"https://es.wikipedia.org/wiki/{urllib.parse.quote(best_title.replace(' ', '_'))}",
                    }
        except Exception:
            return None
        return None


class ScratchVectorRAG:
    """
    Motor Vectorial RAG con caché de dos niveles (SQLite BLOB + Matriz NumPy en RAM).
    Evita recalcular el paso forward del Micro-Transformer para cada fragmento en cada
    consulta, actualizando los vectores únicamente cuando cambian las épocas entrenadas.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._cache_dirty: bool = True
        self._cached_epoch: int = -1
        self._cached_dim: int = -1
        self._cached_entries: List[Dict[str, object]] = []
        self._cached_matrix: Optional[np.ndarray] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunks_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    stems_str TEXT NOT NULL,
                    token_ids_json TEXT DEFAULT '[]',
                    embedding_blob BLOB DEFAULT NULL,
                    embed_epoch INTEGER DEFAULT -1,
                    FOREIGN KEY(doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
                )
                """
            )
            existing_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(rag_chunks)").fetchall()
            }
            if "token_ids_json" not in existing_cols:
                conn.execute("ALTER TABLE rag_chunks ADD COLUMN token_ids_json TEXT DEFAULT '[]'")
            if "embedding_blob" not in existing_cols:
                conn.execute("ALTER TABLE rag_chunks ADD COLUMN embedding_blob BLOB DEFAULT NULL")
            if "embed_epoch" not in existing_cols:
                conn.execute("ALTER TABLE rag_chunks ADD COLUMN embed_epoch INTEGER DEFAULT -1")
            conn.commit()

    def invalidate_vector_cache(self):
        """Marca la caché vectorial como obsoleta tras entrenar nuevas épocas del modelo."""
        with self._lock:
            self._cache_dirty = True
            self._cached_epoch = -1

    def refresh_vector_cache(self, tokenizer, neural_net):
        """Fuerza la sincronización y recálculo en lote de los vectores de todos los fragmentos."""
        with self._lock:
            self._ensure_cache_synced(tokenizer, neural_net, force_recompute=True)

    @staticmethod
    def split_into_chunks(text: str, max_chars: int = 420) -> List[str]:
        text = re.sub(r"\r\n", "\n", text)
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 15]
        chunks: List[str] = []

        for para in paragraphs:
            if len(para) <= max_chars:
                chunks.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                curr = ""
                for s in sentences:
                    if len(curr) + len(s) + 1 <= max_chars:
                        curr = f"{curr} {s}".strip()
                    else:
                        if len(curr) >= 20:
                            chunks.append(curr)
                        curr = s.strip()
                if len(curr) >= 20:
                    chunks.append(curr)

        return chunks[:150]

    @staticmethod
    def _compute_stems_string(text: str) -> str:
        norm = strip_accents(text.lower())
        words = re.findall(r"[a-zñ0-9]{3,}", norm)
        stems = [spanish_stem(w) for w in words if w not in STOPWORDS_ES]
        return " ".join(stems)

    def ingest_document(
        self,
        title: str,
        text: str,
        source_type: str = "file",
        tokenizer=None,
        neural_net=None,
    ) -> Dict[str, object]:
        chunks = self.split_into_chunks(text)
        if not chunks:
            return {"ok": False, "chunks_added": 0, "doc_id": None}

        now = time.time()
        with self._lock:
            chunk_rows = []
            current_epoch = int(neural_net.total_epochs_trained) if neural_net is not None else -1

            for idx, c in enumerate(chunks):
                stems_str = self._compute_stems_string(c)
                tok_ids: List[int] = []
                emb_bytes: Optional[bytes] = None
                ep_tag = -1

                if tokenizer is not None:
                    tok_ids = tokenizer.encode(c[:280], learn_vocab=True)
                    if neural_net is not None:
                        neural_net.expand_vocab(len(tokenizer.word2idx))
                        vec = neural_net.encode_tokens(tok_ids)[0].astype(np.float32)
                        emb_bytes = vec.tobytes()
                        ep_tag = current_epoch

                chunk_rows.append(
                    (
                        idx,
                        c,
                        stems_str,
                        json.dumps(tok_ids),
                        emb_bytes,
                        ep_tag,
                    )
                )

            with self._connect() as conn:
                old = conn.execute(
                    "SELECT id FROM rag_documents WHERE lower(title) = ?", (title.lower(),)
                ).fetchone()
                if old:
                    conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (old[0],))
                    conn.execute("DELETE FROM rag_documents WHERE id = ?", (old[0],))

                cur = conn.execute(
                    "INSERT INTO rag_documents (title, source_type, chunks_count, created_at) VALUES (?, ?, ?, ?)",
                    (title, source_type, len(chunks), now),
                )
                doc_id = cur.lastrowid

                conn.executemany(
                    """
                    INSERT INTO rag_chunks
                    (doc_id, chunk_index, content, stems_str, token_ids_json, embedding_blob, embed_epoch)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(doc_id, *row) for row in chunk_rows],
                )
                conn.commit()

            self._cache_dirty = True

        return {
            "ok": True,
            "doc_id": doc_id,
            "title": title,
            "chunks_added": len(chunks),
            "chunks": chunks,
        }

    def _ensure_cache_synced(self, tokenizer, neural_net, force_recompute: bool = False):
        target_epoch = int(neural_net.total_epochs_trained)
        embed_dim = int(neural_net.embed_dim)

        if (
            not self._cache_dirty
            and not force_recompute
            and self._cached_epoch == target_epoch
            and self._cached_dim == embed_dim
            and self._cached_matrix is not None
        ):
            return

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, d.title, d.source_type, c.content, c.stems_str,
                       c.token_ids_json, c.embedding_blob, c.embed_epoch
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.doc_id
                ORDER BY c.id ASC
                """
            ).fetchall()

            if not rows:
                self._cached_entries = []
                self._cached_matrix = np.zeros((0, embed_dim), dtype=np.float32)
                self._cached_epoch = target_epoch
                self._cached_dim = embed_dim
                self._cache_dirty = False
                return

            entries: List[Dict[str, object]] = []
            vectors: List[np.ndarray] = []
            db_updates = []

            for cid, doc_title, src_type, content, stems_str, tok_json, emb_blob, emb_epoch in rows:
                try:
                    tok_ids = json.loads(tok_json) if tok_json else []
                except Exception:
                    tok_ids = []

                needs_tokenizing = not tok_ids
                if needs_tokenizing:
                    tok_ids = tokenizer.encode(content[:280], learn_vocab=False)

                vec: Optional[np.ndarray] = None
                if (
                    not force_recompute
                    and emb_blob is not None
                    and int(emb_epoch) == target_epoch
                    and len(emb_blob) == embed_dim * 4
                ):
                    vec = np.frombuffer(emb_blob, dtype=np.float32).copy()

                if vec is None:
                    vec = neural_net.encode_tokens(tok_ids)[0].astype(np.float32)
                    db_updates.append(
                        (json.dumps(tok_ids), vec.tobytes(), target_epoch, cid)
                    )

                stems_set: Set[str] = set(stems_str.split())
                title_stems: Set[str] = set(self._compute_stems_string(doc_title).split())
                entries.append(
                    {
                        "chunk_id": cid,
                        "title": doc_title,
                        "source_type": src_type,
                        "content": content,
                        "stems_set": stems_set,
                        "title_stems": title_stems,
                    }
                )
                vectors.append(vec)

            if db_updates:
                conn.executemany(
                    """
                    UPDATE rag_chunks
                    SET token_ids_json = ?, embedding_blob = ?, embed_epoch = ?
                    WHERE id = ?
                    """,
                    db_updates,
                )
                conn.commit()

        self._cached_entries = entries
        self._cached_matrix = np.vstack(vectors).astype(np.float32)
        self._cached_epoch = target_epoch
        self._cached_dim = embed_dim
        self._cache_dirty = False

    def search(
        self,
        query: str,
        tokenizer,
        neural_net,
        top_k: int = 3,
        min_score: float = 0.28,
    ) -> List[Dict[str, object]]:
        q_stems = set(self._compute_stems_string(query).split())
        if not q_stems:
            return []

        with self._lock:
            self._ensure_cache_synced(tokenizer, neural_net)
            if not self._cached_entries or self._cached_matrix is None or self._cached_matrix.shape[0] == 0:
                return []

            q_ids = tokenizer.encode(query, learn_vocab=False)
            q_vec = neural_net.encode_tokens(q_ids)[0]  # (D,)

            # Producto matricial vectorizado instantáneo sobre la caché RAM: (N, D) @ (D,)
            cos_sims = np.clip(self._cached_matrix @ q_vec, 0.0, 1.0)

            scored = []
            for idx, entry in enumerate(self._cached_entries):
                stems_set: Set[str] = entry["stems_set"]  # type: ignore
                if not stems_set:
                    continue
                title_stems: Set[str] = entry["title_stems"]  # type: ignore

                inter = q_stems & stems_set
                title_inter = q_stems & title_stems
                if not inter and not title_inter:
                    continue

                lex_recall = len(inter) / max(1, len(q_stems))
                title_bonus = 0.25 if title_inter else 0.0
                cos_sim = float(cos_sims[idx])

                hybrid_score = (0.55 * lex_recall) + (0.30 * cos_sim) + title_bonus
                if hybrid_score >= min_score:
                    scored.append(
                        {
                            "chunk_id": entry["chunk_id"],
                            "title": entry["title"],
                            "source_type": entry["source_type"],
                            "content": entry["content"],
                            "score": round(float(min(0.99, hybrid_score)), 3),
                        }
                    )

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

    def list_documents(self) -> List[Dict[str, object]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, source_type, chunks_count, created_at FROM rag_documents ORDER BY id DESC LIMIT 50"
            ).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "source_type": r[2],
                "chunks_count": r[3],
                "date": time.strftime("%d/%m %H:%M", time.localtime(r[4])),
            }
            for r in rows
        ]

    def delete_document(self, doc_id: int):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))
            conn.commit()
            self._cache_dirty = True

    def export_state_dict(self) -> dict:
        with self._lock, self._connect() as conn:
            docs = conn.execute(
                "SELECT id, title, source_type, chunks_count, created_at FROM rag_documents ORDER BY id ASC"
            ).fetchall()
            chunks = conn.execute(
                "SELECT doc_id, chunk_index, content, stems_str, token_ids_json FROM rag_chunks ORDER BY id ASC"
            ).fetchall()
        return {
            "documents": [
                {
                    "id": r[0],
                    "title": r[1],
                    "source_type": r[2],
                    "chunks_count": r[3],
                    "created_at": r[4],
                }
                for r in docs
            ],
            "chunks": [
                {
                    "doc_id": c[0],
                    "chunk_index": c[1],
                    "content": c[2],
                    "stems_str": c[3],
                    "token_ids_json": c[4],
                }
                for c in chunks
            ],
        }

    def import_state_dict(self, data: dict):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks")
            conn.execute("DELETE FROM rag_documents")
            for d in data.get("documents", []):
                conn.execute(
                    "INSERT INTO rag_documents (id, title, source_type, chunks_count, created_at) VALUES (?, ?, ?, ?, ?)",
                    (int(d["id"]), str(d["title"]), str(d["source_type"]), int(d["chunks_count"]), float(d["created_at"])),
                )
            for c in data.get("chunks", []):
                conn.execute(
                    """
                    INSERT INTO rag_chunks
                    (doc_id, chunk_index, content, stems_str, token_ids_json, embedding_blob, embed_epoch)
                    VALUES (?, ?, ?, ?, ?, NULL, -1)
                    """,
                    (
                        int(c["doc_id"]),
                        int(c["chunk_index"]),
                        str(c["content"]),
                        str(c["stems_str"]),
                        str(c.get("token_ids_json", "[]")),
                    ),
                )
            conn.commit()
            self._cached_entries = []
            self._cached_matrix = None
            self._cached_epoch = -1
            self._cache_dirty = True

    def clear_all(self):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks")
            conn.execute("DELETE FROM rag_documents")
            conn.commit()
            self._cached_entries = []
            self._cached_matrix = None
            self._cache_dirty = True
