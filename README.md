# NeuroZero 3.0 (Trifo-AI) — Motor Neuro-Simbólico Desde Cero (Python + NumPy Puro)

Motor de Inteligencia Artificial Neuro-Simbólico construido **100% desde cero usando únicamente la librería estándar de Python y `numpy>=1.26.0`** (sin PyTorch, TensorFlow, Scikit-Learn ni APIs de LLMs externos), con sincronización *Thread-Safe* (`RLock` + SQLite WAL), persistencia atómica e interfaz **Cyber-Lab v3**.

## Pilares de Arquitectura v3

1. **Micro-Transformer Multi-Cabeza + LayerNorm (`engine/neural_net.py`)**:
   - **Multi-Head Self-Attention (`H = 4` cabezas, $d_{head} = d_{model}/4$)** con proyecciones entrenables $W_Q, W_K, W_V$ y matriz de salida $W_O$.
   - **Layer Normalization** exacta en NumPy (paso *forward* y Jacobiano analítico en *backward*) tras las conexiones residuales.
   - **Retropropagación Completa de Contexto** hacia los embeddings y cabezas de atención de los turnos conversacionales previos.
   - **Global L2 Norm Gradient Clipping** + optimizador **Adam** desde cero y guardado atómico (`.tmp` + `os.replace`).
2. **Memoria Multi-Turno + Neural-Markov Reranker (`engine/memory.py` & `engine/brain.py`)**:
   - **Buffer Circular de Trabajo (4 turnos)** ponderado por decaimiento temporal exponencial.
   - **Neural-Markov Reranker ($K = 5$)**: muestrea candidatos con bigramas/trigramas y los clasifica por similitud coseno frente al vector latente predicho por el Micro-Transformer.
3. **Caché Vectorial RAG + Analítica Estadística CSV (`engine/knowledge_rag.py`)**:
   - Caché de dos niveles (SQLite BLOB + matriz vectorizada en RAM) sincronizada con las épocas entrenadas para búsquedas instantáneas sin recalcular pasos *forward* innecesarios.
   - Lectura nativa de `.txt`, `.md`, `.docx`, `.py`, `.csv`, `.json`, `.html`, `.pdf` y **Auto-Investigación en Wikipedia**.
4. **Razonamiento con Contradicciones, Backward Chaining y Agente Multi-Herramienta (`engine/reasoning.py` & `engine/tools_agent.py`)**:
   - **Detección de Contradicciones Lógicas** directas y por herencia transitiva al insertar hechos o reglas.
   - **Backward Chaining** para demostrar preguntas `¿Por qué X es Y?` mostrando el árbol completo de inferencia.
   - **Ejecución Combinada de Herramientas** en un solo mensaje, ecuaciones cuadráticas ($ax^2 + bx + c = 0$ con raíces reales y complejas en $\mathbb{C}$), análisis estadístico `.csv` en NumPy y exportación/importación de respaldos completos del cerebro.
5. **Interfaz Cyber-Lab UI/UX v3 (`static/`)**:
   - Estética *Dark Minimalist* (`#050508` / `#0d0f14` / `#10b981`), tipografía `Inter` + `JetBrains Mono` e iconografía 100% vectorial SVG (`stroke-width="1.75"`, cero emojis).
   - Navegación en 3 vistas (**Chat & Agente** `Alt+1`, **Inspector Neuronal en Vivo** `Alt+2`, **Conocimiento & Lab** `Alt+3`) más **Split View redimensionable** (`Alt+S`).
   - Visualizadores Canvas **High-DPI (Retina)** con `ResizeObserver`, curvas Bézier suaves, *glow* radial en activaciones y tooltips interactivos.

## Ejecución Rápida

```bash
pip install -r requirements.txt
python server.py --open
```

O en Windows, ejecuta **`Iniciar_IA.bat`** para abrir la interfaz en `http://127.0.0.1:8090`.
