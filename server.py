"""
Servidor Web Local para NeuroZero 2.0 (Paquete Completo).
Incluye endpoints REST para:
- Chat Híbrido con Micro-Transformer (Self-Attention) y Auto-Enseñanza.
- Subida e indexación RAG de archivos (.txt, .md, .docx, .py, .csv, .pdf).
- Auto-Investigación en Wikipedia.
- Gestión de Tareas, Recordatorios y Notas (Agente Útil).
"""

import os
import sys
import json
import base64
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from engine.brain import ScratchAIEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")

AI_BRAIN = ScratchAIEngine(data_dir=DATA_DIR)


class NeuroZeroRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return
        if parsed.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json({"ok": False, "error": f"JSON inválido: {e}"}, status=400)
            return

        if parsed.path == "/api/chat":
            msg = str(body.get("message", "")).strip()
            result = AI_BRAIN.process_chat_message(msg)
            self._send_json({"ok": True, **result})
            return

        if parsed.path == "/api/upload_doc":
            filename = str(body.get("filename", "documento.txt")).strip()
            b64 = str(body.get("content_base64", ""))
            try:
                raw_bytes = base64.b64decode(b64)
            except Exception as e:
                self._send_json({"ok": False, "message": f"Archivo inválido: {e}"}, status=400)
                return
            res = AI_BRAIN.ingest_document_bytes(filename=filename, raw_bytes=raw_bytes)
            self._send_json(res)
            return

        if parsed.path == "/api/research":
            query = str(body.get("query", "")).strip()
            if not query:
                self._send_json({"ok": False, "message": "Escribe un tema para investigar."}, status=400)
                return
            res = AI_BRAIN.research_topic_online(query)
            self._send_json(res)
            return

        if parsed.path == "/api/delete_doc":
            doc_id = int(body.get("id", -1))
            AI_BRAIN.rag.delete_document(doc_id)
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/tasks":
            action = str(body.get("action", ""))
            if action == "add":
                desc = str(body.get("description", "")).strip()
                if desc:
                    AI_BRAIN.toolkit.add_task(desc)
            elif action == "toggle":
                tid = int(body.get("id", -1))
                done = bool(body.get("done", True))
                AI_BRAIN.toolkit.toggle_task(tid, done=done)
            elif action == "delete":
                tid = int(body.get("id", -1))
                AI_BRAIN.toolkit.delete_task(tid)
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/notes":
            action = str(body.get("action", ""))
            if action == "add":
                content = str(body.get("content", "")).strip()
                if content:
                    AI_BRAIN.toolkit.add_note(content)
            elif action == "delete":
                nid = int(body.get("id", -1))
                AI_BRAIN.toolkit.delete_note(nid)
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/feedback":
            ep_id = body.get("episode_id")
            fb_type = str(body.get("type", "reward"))
            corrected = body.get("corrected_response")
            res = AI_BRAIN.apply_reinforcement(
                episode_id=int(ep_id) if ep_id is not None else None,
                feedback_type=fb_type,
                corrected_response=str(corrected).strip() if corrected else None,
            )
            res["brain_state"] = AI_BRAIN.get_brain_telemetry()
            self._send_json(res)
            return

        if parsed.path == "/api/teach":
            stim = str(body.get("stimulus", "")).strip()
            resp = str(body.get("response", "")).strip()
            if not stim or not resp:
                self._send_json({"ok": False, "message": "Falta estímulo o respuesta."}, status=400)
                return
            ep, loss = AI_BRAIN.learn_association(stim, resp, reward=1.5, source="manual_panel", steps=8)
            AI_BRAIN.save_state()
            self._send_json(
                {
                    "ok": True,
                    "episode": ep.to_dict(),
                    "loss": round(loss, 5),
                    "brain_state": AI_BRAIN.get_brain_telemetry(),
                }
            )
            return

        if parsed.path == "/api/train":
            epochs = max(1, min(int(body.get("epochs", 20)), 100))
            res = AI_BRAIN.train_epochs(epochs=epochs)
            self._send_json(res)
            return

        if parsed.path == "/api/self_teach":
            res = AI_BRAIN.run_autonomous_self_play_step()
            self._send_json(res)
            return

        if parsed.path == "/api/settings":
            if "curiosity_mode" in body:
                AI_BRAIN.curiosity_mode = bool(body["curiosity_mode"])
                if not AI_BRAIN.curiosity_mode:
                    AI_BRAIN.pending_unknown_stimulus = None
            if "auto_self_play" in body:
                AI_BRAIN.auto_self_play = bool(body["auto_self_play"])
            if "auto_web_research" in body:
                AI_BRAIN.auto_web_research = bool(body["auto_web_research"])
            AI_BRAIN.save_state()
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/delete_episode":
            ep_id = int(body.get("id", -1))
            AI_BRAIN.memory.episodes = [ep for ep in AI_BRAIN.memory.episodes if ep.id != ep_id]
            AI_BRAIN.save_state()
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/delete_fact":
            fact_id = int(body.get("id", -1))
            AI_BRAIN.memory.facts = [f for f in AI_BRAIN.memory.facts if int(f.get("id", 0)) != fact_id]
            AI_BRAIN.save_state()
            self._send_json({"ok": True, "brain_state": AI_BRAIN.get_brain_telemetry()})
            return

        if parsed.path == "/api/neighbors":
            word = str(body.get("word", "")).strip()
            neighbors = AI_BRAIN.get_word_neighbors(word)
            self._send_json({"ok": True, "word": word, "neighbors": neighbors})
            return

        if parsed.path == "/api/reset":
            include_seed = bool(body.get("include_seed", True))
            AI_BRAIN.reset_brain(include_seed=include_seed)
            self._send_json(
                {
                    "ok": True,
                    "message": (
                        "Cerebro NeuroZero 2.0 reiniciado con semilla básica."
                        if include_seed
                        else "Cerebro reiniciado 100% Virgen (Tabula Rasa)."
                    ),
                    "brain_state": AI_BRAIN.get_brain_telemetry(),
                }
            )
            return

        self._send_json({"ok": False, "error": "Ruta no encontrada"}, status=404)


def run_server(port: int = 8090, open_browser: bool = True):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, NeuroZeroRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 68)
    print("  NEUROZERO 2.0 - MICRO-TRANSFORMER + RAG + AGENTE + MOTOR LOGICO")
    print("=" * 68)
    print(f"  Servidor activo en: {url}")
    print("=" * 68)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        AI_BRAIN.save_state()
        httpd.server_close()


if __name__ == "__main__":
    run_server(port=8090, open_browser=("--open" in sys.argv))
