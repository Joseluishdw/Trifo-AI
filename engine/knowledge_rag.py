"""
Motor de Super-Aprendizaje Masivo (RAG Vectorial desde Cero + Lector de Archivos + Wikipedia).
Capacidades:
1. Lectura nativa desde cero de .txt, .md, .csv, .py, .json, .html, .docx (vía XML/Zip)
   y extracción básica de .pdf (vía descompresión zlib de streams).
2. Fragmentación semántica (Chunking) y almacenamiento persistente en SQLite.
3. Búsqueda Híbrida (Similitud Coseno en NumPy con Embeddings Transformer + BM25 Morfológico).
4. Auto-Investigación Autónoma en Wikipedia en Español cuando el usuario pide investigar
   un tema o cuando la IA necesita aprender sobre un concepto desconocido.
"""

import io
import os
import re
import json
import zlib
import time
import sqlite3
import zipfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import numpy as np

from .tokenizer import strip_accents, spanish_stem, STOPWORDS_ES


class DocumentParserFromScratch:
    """Extrae texto limpio de múltiples formatos sin librerías externas pesadas."""

    @staticmethod
    def parse_bytes(filename: str, raw_bytes: bytes) -> str:
        ext = os.path.splitext(filename.lower())[1]

        if ext == ".docx":
            return DocumentParserFromScratch._parse_docx(raw_bytes)
        if ext == ".pdf":
            return DocumentParserFromScratch._parse_pdf_basic(raw_bytes)
        if ext in {".html", ".htm"}:
            text = DocumentParserFromScratch._decode_text(raw_bytes)
            text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        # Texto plano, Markdown, CSV, JSON, Código (.py, .js, .java, .sql, etc.)
        return DocumentParserFromScratch._decode_text(raw_bytes)

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
        """Lee archivos Word (.docx) descomprimiendo su XML interno desde cero."""
        paragraphs = []
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml_content = zf.read("word/document.xml")
            root = ET.fromstring(xml_content)
            for elem in root.iter():
                if elem.tag.endswith("}p"):
                    texts = [t.text for t in elem.iter() if t.tag.endswith("}t") and t.text]
                    if texts:
                        paragraphs.append("".join(texts))
        return "\n".join(paragraphs)

    @staticmethod
    def _parse_pdf_basic(raw_bytes: bytes) -> str:
        """
        Extractor ligero de flujos de texto PDF desde cero usando zlib para FlateDecode.
        Funciona con PDFs digitales estándar con texto embebido simple.
        """
        extracted_parts = []
        streams = re.findall(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw_bytes, re.DOTALL)
        for s in streams:
            data = s
            try:
                data = zlib.decompress(s)
            except Exception:
                try:
                    data = zlib.decompress(s.strip())
                except Exception:
                    pass
            # Extraer cadenas entre paréntesis dentro de bloques BT ... ET
            txt_chunks = re.findall(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*Tj", data)
            for tc in txt_chunks:
                decoded = tc.decode("latin-1", errors="ignore")
                extracted_parts.append(decoded)
            arr_chunks = re.findall(rb"\[(.*?)\]\s*TJ", data, re.DOTALL)
            for ac in arr_chunks:
                sub_strs = re.findall(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)", ac)
                if sub_strs:
                    joined = "".join(x.decode("latin-1", errors="ignore") for x in sub_strs)
                    extracted_parts.append(joined)

        result = " ".join(extracted_parts)
        result = re.sub(r"\s+", " ", result).strip()
        return result


class WikipediaAutoResearcher:
    """
    Permite que la IA investigue temas por sí sola en Wikipedia en Español,
    descargando conocimiento real para auto-entrenarse.
    """

    USER_AGENT = "NeuroZeroFromScratch/2.0 (Educational Local AI Project; Python urllib)"

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
            # 1. Buscar el título más relevante en es.wikipedia.org
            search_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_q,
                    "srlimit": 1,
                    "format": "json",
                }
            )
            url_search = f"https://es.wikipedia.org/w/api.php?{search_params}"
            req = urllib.request.Request(url_search, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=7) as resp:
                search_data = json.loads(resp.read().decode("utf-8"))

            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return None

            best_title = hits[0]["title"]

            # 2. Descargar el texto plano limpio del artículo
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
            url_extract = f"https://es.wikipedia.org/w/api.php?{extract_params}"
            req2 = urllib.request.Request(url_extract, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req2, timeout=8) as resp2:
                ext_data = json.loads(resp2.read().decode("utf-8"))

            pages = ext_data.get("query", {}).get("pages", {})
            for _, page_info in pages.items():
                extract = page_info.get("extract", "").strip()
                if extract:
                    # Limpiar encabezados tipo "== Historia =="
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
    Base de Conocimiento Vectorial (RAG) construida desde cero sobre SQLite + NumPy.
    Fragmenta documentos largos y artículos investigados, permitiendo recuperar los
    párrafos exactos combinando BM25 morfológico + Similitud Coseno Neuronal.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
                    FOREIGN KEY(doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    @staticmethod
    def split_into_chunks(text: str, max_chars: int = 420) -> List[str]:
        """Divide un documento largo en fragmentos coherentes respetando oraciones."""
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

        return chunks[:150]  # Límite sano por documento para mantener velocidad instantánea

    @staticmethod
    def _compute_stems_string(text: str) -> str:
        norm = strip_accents(text.lower())
        words = re.findall(r"[a-zñ0-9]{3,}", norm)
        stems = [spanish_stem(w) for w in words if w not in STOPWORDS_ES]
        return " ".join(stems)

    def ingest_document(self, title: str, text: str, source_type: str = "file") -> Dict[str, object]:
        chunks = self.split_into_chunks(text)
        if not chunks:
            return {"ok": False, "chunks_added": 0, "doc_id": None}

        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            # Si ya existía un documento con exactamente el mismo título, reemplazarlo
            old = conn.execute("SELECT id FROM rag_documents WHERE lower(title) = ?", (title.lower(),)).fetchone()
            if old:
                conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (old[0],))
                conn.execute("DELETE FROM rag_documents WHERE id = ?", (old[0],))

            cur = conn.execute(
                "INSERT INTO rag_documents (title, source_type, chunks_count, created_at) VALUES (?, ?, ?, ?)",
                (title, source_type, len(chunks), now),
            )
            doc_id = cur.lastrowid

            chunk_rows = [
                (doc_id, idx, c, self._compute_stems_string(c))
                for idx, c in enumerate(chunks)
            ]
            conn.executemany(
                "INSERT INTO rag_chunks (doc_id, chunk_index, content, stems_str) VALUES (?, ?, ?, ?)",
                chunk_rows,
            )
            conn.commit()

        return {"ok": True, "doc_id": doc_id, "title": title, "chunks_added": len(chunks), "chunks": chunks}

    def search(
        self,
        query: str,
        tokenizer,
        neural_net,
        top_k: int = 3,
        min_score: float = 0.28,
    ) -> List[Dict[str, object]]:
        """
        Búsqueda Híbrida Vectorial + Morfológica (BM25 ligero + Transformer Cosine Similarity).
        """
        q_stems = set(self._compute_stems_string(query).split())
        if not q_stems:
            return []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, d.title, d.source_type, c.content, c.stems_str
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.doc_id
                """
            ).fetchall()

        if not rows:
            return []

        q_ids = tokenizer.encode(query, learn_vocab=False)
        q_vec = neural_net.encode_tokens(q_ids)

        scored = []
        for cid, doc_title, src_type, content, stems_str in rows:
            c_stems_list = stems_str.split()
            c_stems_set = set(c_stems_list)
            if not c_stems_set:
                continue

            # 1. Coincidencia morfológica ponderada
            inter = q_stems & c_stems_set
            title_stems = set(self._compute_stems_string(doc_title).split())
            title_inter = q_stems & title_stems

            if not inter and not title_inter:
                continue

            lex_recall = len(inter) / max(1, len(q_stems))
            title_bonus = 0.25 if title_inter else 0.0

            # 2. Similitud Coseno en el espacio del Micro-Transformer
            c_ids = tokenizer.encode(content[:260], learn_vocab=False)
            c_vec = neural_net.encode_tokens(c_ids)
            cos_sim = float(np.clip(np.sum(q_vec * c_vec), 0.0, 1.0))

            hybrid_score = (0.55 * lex_recall) + (0.30 * cos_sim) + title_bonus
            if hybrid_score >= min_score:
                scored.append(
                    {
                        "chunk_id": cid,
                        "title": doc_title,
                        "source_type": src_type,
                        "content": content,
                        "score": round(float(min(0.99, hybrid_score)), 3),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_documents(self) -> List[Dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))
            conn.commit()

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM rag_chunks")
            conn.execute("DELETE FROM rag_documents")
            conn.commit()
