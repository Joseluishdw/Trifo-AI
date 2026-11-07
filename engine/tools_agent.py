"""
NeuroZero 3.0 — Agente Multi-Herramienta, Álgebra Lineal/Cuadrática y Analizador Estadístico CSV.
Incluye:
- Ejecución encadenada de múltiples herramientas en un mismo mensaje (ej. calcular ecuación
  y guardar automáticamente el resultado en una nota SQLite o en un archivo en `exports/`).
- Solucionador de ecuaciones cuadráticas (ax² + bx + c = d) con discriminante real y complejo,
  y ecuaciones lineales de primer grado.
- Analizador estadístico de archivos CSV con NumPy (media, mediana, desviación estándar, min, max).
- Gestor Thread-Safe (RLock + SQLite WAL) de tareas, notas y exportación de archivos.
"""

import os
import re
import ast
import json
import math
import time
import operator
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tokenizer import strip_accents
from .knowledge_rag import DocumentParserFromScratch


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "raiz": math.sqrt,
    "sin": lambda x: math.sin(math.radians(x)) if abs(x) > 2 * math.pi else math.sin(x),
    "sen": lambda x: math.sin(math.radians(x)) if abs(x) > 2 * math.pi else math.sin(x),
    "cos": lambda x: math.cos(math.radians(x)) if abs(x) > 2 * math.pi else math.cos(x),
    "tan": lambda x: math.tan(math.radians(x)) if abs(x) > 2 * math.pi else math.tan(x),
    "log": math.log10,
    "ln": math.log,
    "abs": abs,
    "round": round,
    "factorial": math.factorial,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _fmt_num(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return str(round(val, 5))


def safe_eval_math(expr: str) -> float:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponente fuera de rango.")
            return SAFE_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
            return SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id.lower() in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id.lower()]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id.lower()
            if fname in SAFE_FUNCTIONS and len(node.args) == 1:
                return float(SAFE_FUNCTIONS[fname](_eval(node.args[0])))
        raise ValueError("Expresión inválida")

    return float(_eval(tree))


def solve_quadratic_or_linear_equation(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Resuelve ecuaciones cuadráticas (ax² + bx + c = d) con discriminante real o complejo,
    así como ecuaciones lineales (bx + c = d).
    """
    norm = strip_accents(raw_text.lower())
    norm = re.sub(r"^(?:resuelve(?:\s+la\s+ecuacion)?|ecuacion|calcula|halla\s+x(?:\s+en)?)\s*:?\s*", "", norm).strip()
    norm = norm.replace("²", "^2").replace("**2", "^2").replace(" ", "")

    if "=" not in norm or "x" not in norm:
        return None

    parts = norm.split("=")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None

    # Validar que ambos lados solo contengan términos algebraicos en x
    if not re.match(r"^[0-9x\.\+\-\^\*]+$", parts[0]) or not re.match(r"^[0-9x\.\+\-\^\*]+$", parts[1]):
        return None

    def _parse_poly_side(side_str: str) -> Optional[Tuple[float, float, float]]:
        s = side_str.replace("*", "")
        if not s.startswith(("+", "-")):
            s = "+" + s
        tokens = re.findall(r"[+-][^+-]+", s)
        if "".join(tokens) != s:
            return None

        a, b, c = 0.0, 0.0, 0.0
        for tok in tokens:
            if "x^2" in tok:
                coef_s = tok.replace("x^2", "")
                val = 1.0 if coef_s == "+" else (-1.0 if coef_s == "-" else float(coef_s))
                a += val
            elif "x" in tok:
                coef_s = tok.replace("x", "")
                val = 1.0 if coef_s == "+" else (-1.0 if coef_s == "-" else float(coef_s))
                b += val
            else:
                c += float(tok)
        return a, b, c

    try:
        lhs = _parse_poly_side(parts[0])
        rhs = _parse_poly_side(parts[1])
    except ValueError:
        return None

    if lhs is None or rhs is None:
        return None

    a = lhs[0] - rhs[0]
    b = lhs[1] - rhs[1]
    c = lhs[2] - rhs[2]

    # Caso 1: Ecuación Cuadrática (a != 0)
    if abs(a) > 1e-9:
        disc = (b ** 2) - (4.0 * a * c)
        eq_fmt = f"{_fmt_num(a)}x² + ({_fmt_num(b)})x + ({_fmt_num(c)}) = 0"
        if disc > 1e-9:
            sqrt_d = math.sqrt(disc)
            x1 = (-b + sqrt_d) / (2.0 * a)
            x2 = (-b - sqrt_d) / (2.0 * a)
            res_short = f"x1 = {_fmt_num(x1)}, x2 = {_fmt_num(x2)}"
            return {
                "tool_name": "Solucionador Cuadrático",
                "value_summary": res_short,
                "reply": (
                    f"📐 **Ecuación Cuadrática (`{eq_fmt}`):**\n"
                    f"• Discriminante real positivo: `Δ = b² - 4ac = {_fmt_num(disc)} > 0`\n"
                    f"• Dos raíces reales: **`x₁ = {_fmt_num(x1)}`** y **`x₂ = {_fmt_num(x2)}`**"
                ),
            }
        elif abs(disc) <= 1e-9:
            x0 = -b / (2.0 * a)
            res_short = f"x = {_fmt_num(x0)} (raíz doble)"
            return {
                "tool_name": "Solucionador Cuadrático",
                "value_summary": res_short,
                "reply": (
                    f"📐 **Ecuación Cuadrática (`{eq_fmt}`):**\n"
                    f"• Discriminante nulo: `Δ = 0`\n"
                    f"• Raíz real única doble: **`x = {_fmt_num(x0)}`**"
                ),
            }
        else:
            real_p = -b / (2.0 * a)
            imag_p = math.sqrt(abs(disc)) / abs(2.0 * a)
            r1 = f"{_fmt_num(real_p)} + {_fmt_num(imag_p)}i"
            r2 = f"{_fmt_num(real_p)} - {_fmt_num(imag_p)}i"
            res_short = f"x1 = {r1}, x2 = {r2}"
            return {
                "tool_name": "Solucionador Cuadrático (Complejo)",
                "value_summary": res_short,
                "reply": (
                    f"📐 **Ecuación Cuadrática (`{eq_fmt}`):**\n"
                    f"• Discriminante negativo: `Δ = {_fmt_num(disc)} < 0` (raíces complejas conjugadas)\n"
                    f"• Solución en ℂ: **`x₁ = {r1}`** y **`x₂ = {r2}`**"
                ),
            }

    # Caso 2: Ecuación Lineal de 1er Grado (a == 0, b != 0)
    if abs(b) > 1e-9:
        x_val = -c / b
        res_short = f"x = {_fmt_num(x_val)}"
        return {
            "tool_name": "Solucionador Algebraico Lineal",
            "value_summary": res_short,
            "reply": (
                f"📐 **Solución de la ecuación lineal:**\n"
                f"• Forma reducida: `{_fmt_num(b)}x = {_fmt_num(-c)}`\n"
                f"• Resultado exacto: **`x = {_fmt_num(x_val)}`**"
            ),
        }

    return None


class ScratchAgentToolkit:
    def __init__(self, db_path: str, exports_dir: str):
        self.db_path = db_path
        self.exports_dir = exports_dir
        self._lock = threading.RLock()
        os.makedirs(self.exports_dir, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    done INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS csv_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def register_csv_dataset(self, filename: str, raw_bytes: bytes) -> Optional[Dict[str, object]]:
        stats = DocumentParserFromScratch.analyze_csv_statistics(raw_bytes)
        if not stats:
            return None
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM csv_datasets WHERE lower(filename) = ?", (filename.lower(),))
            conn.execute(
                "INSERT INTO csv_datasets (filename, stats_json, created_at) VALUES (?, ?, ?)",
                (filename, json.dumps(stats, ensure_ascii=False), now),
            )
            conn.commit()
        return stats

    def format_csv_stats_report(self, filename: str, stats: Dict[str, object]) -> str:
        lines = [
            f"📊 **Análisis Estadístico NumPy de `{filename}`** ({stats['rows_count']} filas, {len(stats['columns'])} columnas):"
        ]
        num_stats: Dict[str, Dict[str, float]] = stats.get("numeric_stats", {})  # type: ignore
        if not num_stats:
            lines.append("• No se detectaron columnas numéricas con suficientes datos.")
        else:
            for col, st in num_stats.items():
                lines.append(
                    f"• **`{col}`** *(n={st['count']})*: Media = **{st['mean']}** | Mediana = **{st['median']}** | "
                    f"Desv. Est. (σ) = **{st['std']}** | Mín = **{st['min']}** | Máx = **{st['max']}**"
                )
        return "\n".join(lines)

    def query_csv_statistics(self, user_text: str) -> Optional[Dict[str, str]]:
        norm = strip_accents(user_text.lower())
        csv_triggers = (
            "estadistica", "media", "mediana", "desviacion", "promedio",
            "maximo", "minimo", "analiza el csv", "datos del csv", ".csv"
        )
        if not any(t in norm for t in csv_triggers):
            return None

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT filename, stats_json FROM csv_datasets ORDER BY id DESC LIMIT 10"
            ).fetchall()

        if not rows:
            return None

        chosen_file, chosen_json = rows[0]
        for fname, sjson in rows:
            if strip_accents(fname.lower()) in norm or os.path.splitext(strip_accents(fname.lower()))[0] in norm:
                chosen_file, chosen_json = fname, sjson
                break

        stats = json.loads(chosen_json)
        report = self.format_csv_stats_report(chosen_file, stats)
        return {
            "tool_name": "Analizador Estadístico CSV (NumPy)",
            "value_summary": report,
            "reply": report,
        }

    def add_task(self, description: str) -> Dict[str, object]:
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (description, done, created_at) VALUES (?, 0, ?)",
                (description.strip(), now),
            )
            tid = cur.lastrowid
            conn.commit()
        return {"id": tid, "description": description.strip(), "done": False}

    def toggle_task(self, task_id: int, done: bool = True) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("UPDATE tasks SET done = ? WHERE id = ?", (1 if done else 0, task_id))
            conn.commit()
            return cur.rowcount > 0

    def delete_task(self, task_id: int):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()

    def list_tasks(self, only_pending: bool = False) -> List[Dict[str, object]]:
        with self._lock, self._connect() as conn:
            if only_pending:
                rows = conn.execute(
                    "SELECT id, description, done, created_at FROM tasks WHERE done = 0 ORDER BY id DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, description, done, created_at FROM tasks ORDER BY done ASC, id DESC LIMIT 50"
                ).fetchall()
        return [
            {
                "id": r[0],
                "description": r[1],
                "done": bool(r[2]),
                "date": time.strftime("%d/%m %H:%M", time.localtime(r[3])),
            }
            for r in rows
        ]

    def add_note(self, content: str) -> Dict[str, object]:
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute("INSERT INTO notes (content, created_at) VALUES (?, ?)", (content.strip(), now))
            nid = cur.lastrowid
            conn.commit()
        return {"id": nid, "content": content.strip()}

    def delete_note(self, note_id: int):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()

    def list_notes(self, query: str = "") -> List[Dict[str, object]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, content, created_at FROM notes ORDER BY id DESC LIMIT 50").fetchall()
        notes = [
            {
                "id": r[0],
                "content": r[1],
                "date": time.strftime("%d/%m %H:%M", time.localtime(r[2])),
            }
            for r in rows
        ]
        if query:
            q_norm = strip_accents(query.lower().strip())
            notes = [n for n in notes if q_norm in strip_accents(str(n["content"]).lower())]
        return notes

    def export_state_dict(self) -> dict:
        with self._lock, self._connect() as conn:
            tasks = conn.execute("SELECT id, description, done, created_at FROM tasks").fetchall()
            notes = conn.execute("SELECT id, content, created_at FROM notes").fetchall()
            csvs = conn.execute("SELECT filename, stats_json, created_at FROM csv_datasets").fetchall()
        return {
            "tasks": [{"id": r[0], "description": r[1], "done": r[2], "created_at": r[3]} for r in tasks],
            "notes": [{"id": r[0], "content": r[1], "created_at": r[2]} for r in notes],
            "csv_datasets": [{"filename": r[0], "stats_json": r[1], "created_at": r[2]} for r in csvs],
        }

    def import_state_dict(self, data: dict):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM csv_datasets")
            for t in data.get("tasks", []):
                conn.execute(
                    "INSERT INTO tasks (id, description, done, created_at) VALUES (?, ?, ?, ?)",
                    (t["id"], t["description"], int(bool(t["done"])), float(t["created_at"])),
                )
            for n in data.get("notes", []):
                conn.execute(
                    "INSERT INTO notes (id, content, created_at) VALUES (?, ?, ?)",
                    (n["id"], n["content"], float(n["created_at"])),
                )
            for c in data.get("csv_datasets", []):
                conn.execute(
                    "INSERT INTO csv_datasets (filename, stats_json, created_at) VALUES (?, ?, ?)",
                    (c["filename"], c["stats_json"], float(c["created_at"])),
                )
            conn.commit()

    def clear_all(self):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM csv_datasets")
            conn.commit()

    # ==========================================================
    # PIPELINE MULTI-HERRAMIENTA Y ACCIONES ENCADENADAS
    # ==========================================================
    def _split_multi_tool_clauses(self, text: str) -> List[str]:
        """
        Divide instrucciones compuestas para permitir ejecución encadenada:
        ej. 'Calcula 25 * 16 y guárdalo en una nota' o 'Resuelve x^2 - 9 = 0 y guárdalo en raices.txt'.
        """
        pattern = (
            r"\s+(?:;\s*|y\s+luego\s+|y\s+ademas\s+|"
            r"y\s+(?=(?:guardalo|guarda(?:\s+el\s+resultado)?|anotalo|agrega(?:r)?\s+tarea|recuerdame|crea(?:r)?\s+(?:un\s+)?archivo|calcula|resuelve|cuanto\s+es)))"
        )
        # Usar versión sin acentos para localizar cortes exactos manteniendo caracteres originales
        norm_text = strip_accents(text)
        splits = []
        last_idx = 0
        for m in re.finditer(pattern, norm_text, flags=re.IGNORECASE):
            part = text[last_idx : m.start()].strip()
            if part:
                splits.append(part)
            last_idx = m.end()
        tail = text[last_idx:].strip()
        if tail:
            splits.append(tail)
        return splits if len(splits) > 1 else [text]

    def try_execute_tool(self, user_text: str) -> Optional[Dict[str, object]]:
        clauses = self._split_multi_tool_clauses(user_text.strip())
        if len(clauses) > 1:
            executed = []
            prev_val: Optional[str] = None
            for clause in clauses:
                res = self._execute_single_tool(clause, prev_result=prev_val)
                if res is None:
                    # Si una cláusula no era una herramienta, abortar modo multi-herramienta salvo que ya hubiera éxito claro
                    if not executed:
                        return None
                    continue
                executed.append(res)
                if res.get("value_summary"):
                    prev_val = str(res["value_summary"])

            if len(executed) >= 2:
                tool_names = " + ".join(str(e["tool_name"]) for e in executed)
                combined_reply = (
                    f"⛓️ **Ejecución Multi-Herramienta ({len(executed)} pasos completados):**\n\n"
                    + "\n\n".join(f"**Paso {i+1}:** {e['reply']}" for i, e in enumerate(executed))
                )
                return {
                    "tool_name": f"Pipeline ({tool_names})",
                    "reply": combined_reply,
                }
            if len(executed) == 1:
                return executed[0]
            return None

        return self._execute_single_tool(user_text.strip(), prev_result=None)

    def _execute_single_tool(
        self, raw_clause: str, prev_result: Optional[str] = None
    ) -> Optional[Dict[str, object]]:
        raw = raw_clause.strip()
        clean = re.sub(r"[?¿!¡]+$", "", raw).strip()
        norm = strip_accents(clean.lower())

        # 0. Acciones encadenadas que consumen `prev_result` ("guárdalo en una nota" / "guarda el resultado en X.txt")
        if prev_result is not None:
            m_save_file_ref = re.match(
                r"(?i)^(?:guardalo|guarda(?:\s+el\s+resultado)?|exportalo)\s+(?:en\s+)?(?:el\s+)?(?:archivo\s+)?([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]{1,5})$",
                norm,
            )
            if m_save_file_ref:
                safe_name = os.path.basename(m_save_file_ref.group(1).strip())
                out_path = os.path.join(self.exports_dir, safe_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"Resultado generado por NeuroZero 3.0:\n{prev_result}\n")
                return {
                    "tool_name": "Exportador de Archivos",
                    "value_summary": prev_result,
                    "reply": f"💾 Resultado (`{prev_result}`) guardado en `exports/{safe_name}`.",
                }

            if re.match(
                r"(?i)^(?:guardalo|guarda(?:\s+el\s+resultado)?|anotalo)\s+en\s+(?:una\s+|mis\s+)?notas?$",
                norm,
            ):
                n = self.add_note(f"Resultado calculado: {prev_result}")
                return {
                    "tool_name": "Bloc de Notas Persistente",
                    "value_summary": prev_result,
                    "reply": f"📝 Resultado guardado en la **Nota #{n['id']}**: «Resultado calculado: {prev_result}».",
                }

            if re.match(r"(?i)^(?:agrega(?:lo)?|crea)\s+(?:como\s+)?tarea", norm):
                t = self.add_task(f"Revisar resultado: {prev_result}")
                return {
                    "tool_name": "Gestor de Tareas",
                    "value_summary": prev_result,
                    "reply": f"✅ Creada **Tarea #{t['id']}** con el resultado (`{prev_result}`).",
                }

        # 1. CONSULTA ESTADÍSTICA DE DATASETS CSV
        csv_ans = self.query_csv_statistics(raw)
        if csv_ans is not None:
            return csv_ans

        # 2. FECHA Y HORA ACTUAL
        if any(
            p in norm
            for p in [
                "que hora es", "dime la hora", "hora actual",
                "que dia es hoy", "que fecha es hoy", "fecha actual", "en que dia estamos"
            ]
        ):
            now = datetime.now()
            dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            meses = [
                "enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
            ]
            dt_str = f"{now.strftime('%H:%M:%S')} del {dias[now.weekday()]} {now.day} de {meses[now.month - 1]} de {now.year}"
            return {
                "tool_name": "Reloj del Sistema",
                "value_summary": dt_str,
                "reply": f"🕒 Son las **{dt_str}**.",
            }

        # 3. TAREAS Y RECORDATORIOS
        if re.match(
            r"(?i)^(?:agrega(?:r)?\s+(?:una\s+)?tarea|anade\s+tarea|nueva\s+tarea|recuerdame|anota\s+pendiente|agrega\s+a\s+mis\s+pendientes)\s*:?\s+(.+)$",
            strip_accents(raw),
        ):
            desc = re.sub(
                r"(?i)^(?:agrega(?:r)?\s+(?:una\s+)?tarea|añade\s+tarea|anade\s+tarea|nueva\s+tarea|recuérdame|recuerdame|anota\s+pendiente|agrega\s+a\s+mis\s+pendientes)\s*:?\s+",
                "",
                raw,
            ).strip()
            if prev_result:
                desc = desc.replace("ese resultado", prev_result).replace("el resultado", prev_result)
            t = self.add_task(desc)
            return {
                "tool_name": "Gestor de Tareas",
                "value_summary": desc,
                "reply": f"✅ **Tarea #{t['id']} guardada:** «{t['description']}».",
            }

        m_task_done = re.match(
            r"(?i)^(?:completa(?:r)?|termina(?:r)?|marca(?:r)?(?:\s+como\s+lista)?)\s+(?:la\s+)?tarea\s+#?(\d+)$",
            norm,
        )
        if m_task_done:
            tid = int(m_task_done.group(1))
            ok = self.toggle_task(tid, done=True)
            msg = f"🎉 **Tarea #{tid}** marcada como completada." if ok else f"No existe la tarea #{tid}."
            return {"tool_name": "Gestor de Tareas", "value_summary": f"Tarea #{tid}", "reply": msg}

        if any(
            p in norm
            for p in [
                "que tareas tengo", "mis tareas", "que pendientes tengo",
                "mis pendientes", "ver tareas", "ver mis recordatorios", "lista de tareas"
            ]
        ):
            pending = self.list_tasks(only_pending=True)
            if not pending:
                return {
                    "tool_name": "Gestor de Tareas",
                    "reply": "✨ ¡No tienes tareas pendientes!",
                }
            lines = [f"• **[#{t['id']}]** {t['description']} *({t['date']})*" for t in pending]
            return {
                "tool_name": "Gestor de Tareas",
                "reply": f"📋 **Tienes {len(pending)} tarea(s) pendiente(s):**\n" + "\n".join(lines),
            }

        # 4. NOTAS PERSONALES
        if re.match(r"(?i)^(?:guarda(?:r)?\s+(?:una\s+)?nota|nueva\s+nota|anota\s+esto|toma\s+nota)\s*:?\s+(.+)$", norm):
            note_body = re.sub(
                r"(?i)^(?:guarda(?:r)?\s+(?:una\s+)?nota|nueva\s+nota|anota\s+esto|toma\s+nota)\s*:?\s+",
                "",
                raw,
            ).strip()
            if prev_result:
                note_body = note_body.replace("el resultado", prev_result)
            n = self.add_note(note_body)
            return {
                "tool_name": "Bloc de Notas",
                "value_summary": note_body,
                "reply": f"📝 **Nota #{n['id']} guardada:** «{n['content']}».",
            }

        m_note_search = re.match(
            r"(?i)^(?:busca(?:r)?\s+en\s+(?:mis\s+)?notas|ver\s+(?:mis\s+)?notas|mis\s+notas)(?:\s+(.+))?$",
            norm,
        )
        if m_note_search:
            q_note = (m_note_search.group(1) or "").strip()
            found = self.list_notes(query=q_note)
            if not found:
                return {"tool_name": "Bloc de Notas", "reply": "📭 No se encontraron notas."}
            lines = [f"• **[Nota #{n['id']}]** {n['content']}" for n in found[:8]]
            return {"tool_name": "Bloc de Notas", "reply": "📒 **Tus notas:**\n" + "\n".join(lines)}

        # 5. CREACIÓN DE ARCHIVOS EN EXPORTS/
        m_file = re.match(
            r"(?i)^(?:crea(?:r)?|guarda(?:r)?|genera(?:r)?)\s+(?:un\s+)?archivo\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]{1,5})\s+(?:con(?:\s+el\s+contenido)?|que\s+diga)\s*:?\s*(.+)$",
            raw,
            flags=re.DOTALL,
        )
        if m_file:
            safe_name = os.path.basename(m_file.group(1).strip())
            file_body = m_file.group(2).strip()
            if prev_result:
                file_body = file_body.replace("{resultado}", prev_result)
            out_path = os.path.join(self.exports_dir, safe_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(file_body)
            return {
                "tool_name": "Exportador de Archivos",
                "value_summary": safe_name,
                "reply": f"💾 **Archivo `{safe_name}` creado** en `exports/{safe_name}` ({len(file_body)} caracteres).",
            }

        # 6. PORCENTAJES
        m_pct = re.search(r"(?i)(\d+(?:\.\d+)?)\s*%\s+de\s+(\d+(?:\.\d+)?)", norm)
        if m_pct:
            pct = float(m_pct.group(1))
            total = float(m_pct.group(2))
            res = (pct / 100.0) * total
            res_fmt = _fmt_num(res)
            return {
                "tool_name": "Calculadora de Porcentajes",
                "value_summary": res_fmt,
                "reply": f"🧮 El **{pct}%** de **{total}** es **{res_fmt}**.",
            }

        # 7. ECUACIONES CUADRÁTICAS (ax² + bx + c = d) Y LINEALES (bx + c = d)
        eq_res = solve_quadratic_or_linear_equation(raw)
        if eq_res is not None:
            return eq_res

        # 8. CONVERSIÓN DE UNIDADES
        conv_res = self._try_unit_conversion(norm)
        if conv_res is not None:
            return {"tool_name": "Conversor de Unidades", "value_summary": conv_res, "reply": conv_res}

        # 9. EXPRESIONES ARITMÉTICAS Y CIENTÍFICAS (AST)
        math_candidate = self._extract_math_expression(norm)
        if math_candidate:
            try:
                val = safe_eval_math(math_candidate)
                val_fmt = _fmt_num(val)
                return {
                    "tool_name": "Calculadora Científica (AST)",
                    "value_summary": val_fmt,
                    "reply": f"🧮 Resultado exacto de `{math_candidate}` = **{val_fmt}**",
                }
            except Exception:
                pass

        return None

    @staticmethod
    def _try_unit_conversion(norm: str) -> Optional[str]:
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(km|kilometros?|millas?|kg|kilos?|kilogramos?|lb|libras?|c|celsius|centigrados|f|fahrenheit|m|metros?|ft|pies?)\s+(?:a|en)\s+(km|kilometros?|millas?|kg|kilos?|kilogramos?|lb|libras?|c|celsius|centigrados|f|fahrenheit|m|metros?|ft|pies?)\b",
            norm,
        )
        if not m:
            return None
        val = float(m.group(1))
        u1, u2 = m.group(2), m.group(3)

        if u1.startswith(("km", "kilom")) and u2.startswith("milla"):
            return f"📏 **{val} km** = **{round(val * 0.621371, 4)} millas**."
        if u1.startswith("milla") and u2.startswith(("km", "kilom")):
            return f"📏 **{val} millas** = **{round(val * 1.60934, 4)} km**."
        if u1.startswith(("kg", "kilo")) and u2.startswith(("lb", "libra")):
            return f"⚖️ **{val} kg** = **{round(val * 2.20462, 4)} libras**."
        if u1.startswith(("lb", "libra")) and u2.startswith(("kg", "kilo")):
            return f"⚖️ **{val} libras** = **{round(val / 2.20462, 4)} kg**."
        if u1 in ("c", "celsius", "centigrados") and u2 in ("f", "fahrenheit"):
            return f"🌡️ **{val} °C** = **{round((val * 9 / 5) + 32, 2)} °F**."
        if u1 in ("f", "fahrenheit") and u2 in ("c", "celsius", "centigrados"):
            return f"🌡️ **{val} °F** = **{round((val - 32) * 5 / 9, 2)} °C**."
        if u1.startswith("m") and not u1.startswith("mi") and u2.startswith(("ft", "pie")):
            return f"📏 **{val} metros** = **{round(val * 3.28084, 4)} pies**."
        if u1.startswith(("ft", "pie")) and u2.startswith("m"):
            return f"📏 **{val} pies** = **{round(val / 3.28084, 4)} metros**."
        return None

    @staticmethod
    def _extract_math_expression(norm: str) -> Optional[str]:
        expr = norm
        replacements = [
            (r"^(?:calcula(?:r)?|cuanto\s+es|resuelve|evalua)\s*:?\s*", ""),
            (r"\braiz\s+cuadrada\s+de\s+(\d+(?:\.\d+)?)", r"sqrt(\1)"),
            (r"\bmas\b", "+"),
            (r"\bmenos\b", "-"),
            (r"\bpor\b", "*"),
            (r"\bentre\b", "/"),
            (r"\bdividido\s+(?:entre|por)\b", "/"),
            (r"\belevado\s+a(?:\s+la)?\b", "**"),
        ]
        for pat, rep in replacements:
            expr = re.sub(pat, rep, expr).strip()

        if not re.match(r"^[0-9\s\.\+\-\*\/\%\(\)\^a-z]+$", expr):
            return None
        has_digit = any(ch.isdigit() for ch in expr)
        has_op = any(
            op in expr
            for op in ("+", "-", "*", "/", "^", "%", "sqrt", "raiz", "sin", "sen", "cos", "tan", "log", "ln", "factorial")
        )
        return expr if (has_digit and has_op) else None
