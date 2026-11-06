"""
Motor de Agente y Herramientas Reales desde Cero (Tool-Calling Engine).
Convierte a la IA en un asistente útil capaz de ejecutar acciones exactas:
1. Motor Matemático Seguro (AST), Trigonometría, Porcentajes, Ecuaciones 1er grado y Conversiones.
2. Gestor Persistente de Tareas / Recordatorios y Notas Personales en SQLite.
3. Creador de Archivos de Texto/Código en carpeta segura `exports/` y Utilidades de Fecha/Hora.
"""

import os
import re
import ast
import math
import time
import operator
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .tokenizer import strip_accents


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


def safe_eval_math(expr: str) -> float:
    """Evalúa expresiones matemáticas usando el árbol AST de Python de forma 100% segura."""
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
                raise ValueError("Exponente demasiado grande.")
            return SAFE_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
            return SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id.lower() in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id.lower()]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id.lower()
            if fname in SAFE_FUNCTIONS and len(node.args) == 1:
                return float(SAFE_FUNCTIONS[fname](_eval(node.args[0])))
        raise ValueError("Expresión no matemática")

    return float(_eval(tree))


class ScratchAgentToolkit:
    """
    Gestiona herramientas prácticas y productividad personal en SQLite.
    """

    def __init__(self, db_path: str, exports_dir: str):
        self.db_path = db_path
        self.exports_dir = exports_dir
        os.makedirs(self.exports_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
            conn.commit()

    # ==========================================================
    # API DE TAREAS Y NOTAS (Para chat e interfaz web)
    # ==========================================================
    def add_task(self, description: str) -> Dict[str, object]:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO tasks (description, done, created_at) VALUES (?, 0, ?)",
                (description.strip(), now),
            )
            tid = cur.lastrowid
            conn.commit()
        return {"id": tid, "description": description.strip(), "done": False}

    def toggle_task(self, task_id: int, done: bool = True) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("UPDATE tasks SET done = ? WHERE id = ?", (1 if done else 0, task_id))
            conn.commit()
            return cur.rowcount > 0

    def delete_task(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()

    def list_tasks(self, only_pending: bool = False) -> List[Dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
            if only_pending:
                rows = conn.execute("SELECT id, description, done, created_at FROM tasks WHERE done = 0 ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute("SELECT id, description, done, created_at FROM tasks ORDER BY done ASC, id DESC LIMIT 50").fetchall()
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
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("INSERT INTO notes (content, created_at) VALUES (?, ?)", (content.strip(), now))
            nid = cur.lastrowid
            conn.commit()
        return {"id": nid, "content": content.strip()}

    def delete_note(self, note_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()

    def list_notes(self, query: str = "") -> List[Dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
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

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM notes")
            conn.commit()

    # ==========================================================
    # DETECTOR Y EJECUTOR DE HERRAMIENTAS EN LENGUAJE NATURAL
    # ==========================================================
    def try_execute_tool(self, user_text: str) -> Optional[Dict[str, object]]:
        """
        Analiza el mensaje del usuario. Si corresponde a una acción/herramienta del Agente,
        la ejecuta y devuelve {"tool_name": ..., "reply": ...}. Si no, retorna None.
        """
        raw = user_text.strip()
        clean = re.sub(r"[?¿!¡]+$", "", raw).strip()
        norm = strip_accents(clean.lower())

        # 1. FECHA Y HORA ACTUAL
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
            dia_sem = dias[now.weekday()]
            mes_nom = meses[now.month - 1]
            return {
                "tool_name": "Reloj y Calendario del Sistema",
                "reply": f"🕒 Son las **{now.strftime('%H:%M:%S')}** del **{dia_sem} {now.day} de {mes_nom} de {now.year}**.",
            }

        # 2. GESTIÓN DE TAREAS Y RECORDATORIOS
        m_task_add = re.match(
            r"(?i)^(?:agrega(?:r)?\s+(?:una\s+)?tarea|anade\s+tarea|nueva\s+tarea|recuerdame|anota\s+pendiente|agrega\s+a\s+mis\s+pendientes)\s*:?\s+(.+)$",
            strip_accents(raw),
        )
        if m_task_add:
            # Extraer texto original conservando acentos desde la misma posición aproximada
            desc = re.sub(
                r"(?i)^(?:agrega(?:r)?\s+(?:una\s+)?tarea|añade\s+tarea|anade\s+tarea|nueva\s+tarea|recuérdame|recuerdame|anota\s+pendiente|agrega\s+a\s+mis\s+pendientes)\s*:?\s+",
                "",
                raw,
            ).strip()
            t = self.add_task(desc)
            return {
                "tool_name": "Gestor de Tareas (SQLite)",
                "reply": f"✅ **Tarea #{t['id']} guardada en tu agenda:** «{t['description']}».\nPuedes verla o marcarla en la pestaña *Tareas & Notas*.",
            }

        m_task_done = re.match(r"(?i)^(?:completa(?:r)?|termina(?:r)?|marca(?:r)?(?:\s+como\s+lista)?)\s+(?:la\s+)?tarea\s+#?(\d+)$", norm)
        if m_task_done:
            tid = int(m_task_done.group(1))
            ok = self.toggle_task(tid, done=True)
            msg = f"🎉 ¡Excelente! Marqué la **Tarea #{tid}** como completada." if ok else f"No encontré ninguna tarea con el ID #{tid}."
            return {"tool_name": "Gestor de Tareas (SQLite)", "reply": msg}

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
                    "tool_name": "Gestor de Tareas (SQLite)",
                    "reply": "✨ ¡No tienes tareas pendientes en este momento! Puedes agregar una diciendo: *«Recuérdame estudiar grafos mañana»*.",
                }
            lines = [f"• **[#{t['id']}]** {t['description']} *(creada {t['date']})*" for t in pending]
            return {
                "tool_name": "Gestor de Tareas (SQLite)",
                "reply": f"📋 **Tienes {len(pending)} tarea(s) pendiente(s):**\n" + "\n".join(lines),
            }

        # 3. GESTIÓN DE NOTAS PERSONALES
        if re.match(r"(?i)^(?:guarda(?:r)?\s+(?:una\s+)?nota|nueva\s+nota|anota\s+esto|toma\s+nota)\s*:?\s+(.+)$", norm):
            note_body = re.sub(
                r"(?i)^(?:guarda(?:r)?\s+(?:una\s+)?nota|nueva\s+nota|anota\s+esto|toma\s+nota)\s*:?\s+",
                "",
                raw,
            ).strip()
            n = self.add_note(note_body)
            return {
                "tool_name": "Bloc de Notas Persistente",
                "reply": f"📝 **Nota #{n['id']} guardada:** «{n['content']}».",
            }

        m_note_search = re.match(r"(?i)^(?:busca(?:r)?\s+en\s+(?:mis\s+)?notas|ver\s+(?:mis\s+)?notas|mis\s+notas)(?:\s+(.+))?$", norm)
        if m_note_search:
            q_note = (m_note_search.group(1) or "").strip()
            found = self.list_notes(query=q_note)
            if not found:
                return {
                    "tool_name": "Bloc de Notas Persistente",
                    "reply": "📭 No encontré notas guardadas con ese criterio. Usa *«Guarda nota: ...»* para crear una.",
                }
            lines = [f"• **[Nota #{n['id']}]** {n['content']}" for n in found[:8]]
            return {
                "tool_name": "Bloc de Notas Persistente",
                "reply": "📒 **Tus notas encontradas:**\n" + "\n".join(lines),
            }

        # 4. CREACIÓN / EXPORTACIÓN DE ARCHIVOS LOCALES
        m_file = re.match(
            r"(?i)^(?:crea(?:r)?|guarda(?:r)?|genera(?:r)?)\s+(?:un\s+)?archivo\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]{1,5})\s+(?:con(?:\s+el\s+contenido)?|que\s+diga)\s*:?\s*(.+)$",
            raw,
            flags=re.DOTALL,
        )
        if m_file:
            safe_name = os.path.basename(m_file.group(1).strip())
            file_body = m_file.group(2).strip()
            out_path = os.path.join(self.exports_dir, safe_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(file_body)
            return {
                "tool_name": "Generador de Archivos Local",
                "reply": f"💾 **Archivo `{safe_name}` creado con éxito** en `exports/{safe_name}` ({len(file_body)} caracteres).",
            }

        # 5. MATEMÁTICAS: PORCENTAJES ("cuanto es el 18% de 4500")
        m_pct = re.search(r"(?i)(\d+(?:\.\d+)?)\s*%\s+de\s+(\d+(?:\.\d+)?)", norm)
        if m_pct:
            pct = float(m_pct.group(1))
            total = float(m_pct.group(2))
            res = (pct / 100.0) * total
            res_fmt = int(res) if res.is_integer() else round(res, 4)
            return {
                "tool_name": "Motor Matemático Exacto",
                "reply": f"🧮 El **{pct}%** de **{total}** es **{res_fmt}**.",
            }

        # 6. ÁLGEBRA LINEAL DE 1ER GRADO ("resuelve 3x + 5 = 20" o "2x - 8 = 10")
        m_eq = re.search(
            r"(?i)(?:resuelve|ecuacion|halla\s+x)?\s*([+-]?\d*(?:\.\d+)?)\s*x\s*([+-]\s*\d+(?:\.\d+)?)?\s*=\s*([+-]?\d+(?:\.\d+)?)",
            norm,
        )
        if m_eq and "=" in norm and "x" in norm:
            a_str = m_eq.group(1).strip()
            a = 1.0 if a_str in ("", "+") else (-1.0 if a_str == "-" else float(a_str))
            b_str = (m_eq.group(2) or "0").replace(" ", "")
            b = float(b_str)
            c = float(m_eq.group(3))
            if abs(a) > 1e-9:
                x_val = (c - b) / a
                x_fmt = int(x_val) if x_val.is_integer() else round(x_val, 4)
                return {
                    "tool_name": "Solucionador Algebraico",
                    "reply": (
                        f"📐 **Solución de la ecuación paso a paso:**\n"
                        f"• Despejando: `{a}x = {c} - ({b})` ➔ `{a}x = {c - b}`\n"
                        f"• Resultado exacto: **`x = {x_fmt}`**"
                    ),
                }

        # 7. CONVERSIONES DE UNIDADES COMUNES
        conv_res = self._try_unit_conversion(norm)
        if conv_res is not None:
            return {"tool_name": "Conversor de Unidades", "reply": conv_res}

        # 8. EVALUADOR DE EXPRESIONES MATEMÁTICAS Y CIENTÍFICAS
        math_candidate = self._extract_math_expression(norm)
        if math_candidate:
            try:
                val = safe_eval_math(math_candidate)
                val_fmt = int(val) if val.is_integer() else round(val, 6)
                return {
                    "tool_name": "Motor Matemático Exacto (AST)",
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
        u1 = m.group(2)
        u2 = m.group(3)

        if u1.startswith(("km", "kilom")) and u2.startswith("milla"):
            return f"📏 **{val} km** equivalen a **{round(val * 0.621371, 4)} millas**."
        if u1.startswith("milla") and u2.startswith(("km", "kilom")):
            return f"📏 **{val} millas** equivalen a **{round(val * 1.60934, 4)} km**."
        if u1.startswith(("kg", "kilo")) and u2.startswith(("lb", "libra")):
            return f"⚖️ **{val} kg** equivalen a **{round(val * 2.20462, 4)} libras**."
        if u1.startswith(("lb", "libra")) and u2.startswith(("kg", "kilo")):
            return f"⚖️ **{val} libras** equivalen a **{round(val / 2.20462, 4)} kg**."
        if u1 in ("c", "celsius", "centigrados") and u2 in ("f", "fahrenheit"):
            return f"🌡️ **{val} °C** equivalen a **{round((val * 9 / 5) + 32, 2)} °F**."
        if u1 in ("f", "fahrenheit") and u2 in ("c", "celsius", "centigrados"):
            return f"🌡️ **{val} °F** equivalen a **{round((val - 32) * 5 / 9, 2)} °C**."
        if u1.startswith("m") and not u1.startswith("mi") and u2.startswith(("ft", "pie")):
            return f"📏 **{val} metros** equivalen a **{round(val * 3.28084, 4)} pies**."
        if u1.startswith(("ft", "pie")) and u2.startswith("m"):
            return f"📏 **{val} pies** equivalen a **{round(val / 3.28084, 4)} metros**."
        return None

    @staticmethod
    def _extract_math_expression(norm: str) -> Optional[str]:
        expr = norm
        # Traducir palabras comunes en español a operadores matemáticos
        replacements = [
            (r"^(?:calcula(?:r)?|cuanto\s+es|resuelve|evalua)\s+", ""),
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

        # Verificar que contenga al menos un número y un operador o función matemática válida
        allowed_pattern = r"^[0-9\s\.\+\-\*\/\%\(\)\^a-z]+$"
        if not re.match(allowed_pattern, expr):
            return None
        has_digit = any(ch.isdigit() for ch in expr)
        has_op = any(op in expr for op in ("+", "-", "*", "/", "^", "%", "sqrt", "raiz", "sin", "sen", "cos", "tan", "log", "ln", "factorial"))
        if has_digit and has_op:
            return expr
        return None
