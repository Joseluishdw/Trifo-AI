# 🧠 NeuroZero 2.0 — Motor de IA Híbrido Desde Cero (Python + NumPy Puro)

Motor de Inteligencia Artificial Neuro-Simbólico construido **100% desde cero en Python y NumPy puro** (sin usar PyTorch, TensorFlow, Scikit-Learn ni APIs de LLMs externos), con aprendizaje continuo en tiempo real hablando por chat.

## 🚀 Pilares de Arquitectura

1. **⚡ Micro-Transformer con *Self-Attention* (`engine/neural_net.py`)**:
   - Matriz dinámica de Embeddings + Codificación Posicional Sinusoidal.
   - Bloque de Auto-Atención entrenable (`Q`, `K`, `V` + *Attention Pooling*) con visualización palabra por palabra.
   - Capas profundas con activación *Swish*, conexión residual *Tanh* y *Backpropagation* exacto con optimizador **Adam** programado desde cero.
2. **🧠 Auto-Enseñanza Autónoma y Morfología (`engine/tokenizer.py` & `engine/memory.py`)**:
   - Stemmer morfológico en español desde cero, tolerancia a errores con trigramas de caracteres y cambio automático de perspectiva gramatical (1ª ↔ 2ª persona).
   - Auto-generación de pares de entrenamiento (*Self-Teaching*) y consolidación neuronal en segundo plano (*Self-Play*).
3. **📚 Super-Aprendizaje Documental RAG + Wikipedia (`engine/knowledge_rag.py`)**:
   - Lectura nativa de documentos `.txt`, `.md`, `.docx`, `.py`, `.csv`, `.json`, `.html` y `.pdf`.
   - Base vectorial híbrida sobre SQLite + NumPy y **Auto-Investigación en Wikipedia** cuando desconoce un tema.
4. **🧩 Inferencia Lógica y Agente de Herramientas (`engine/reasoning.py` & `engine/tools_agent.py`)**:
   - Silogismos multi-salto y encadenamiento condicional *Modus Ponens*.
   - Calculadora científica segura (AST), solucionador de ecuaciones de 1er grado, conversor de unidades, gestor persistente de tareas/notas en SQLite y creación de archivos.

## 🛠️ Ejecución Rápida

```bash
pip install -r requirements.txt
python server.py --open
```

O en Windows, haz doble clic sobre **`Iniciar_IA.bat`** para abrir el laboratorio interactivo en `http://127.0.0.1:8090`.
