@echo off
chcp 65001 >nul
title NeuroZero 3.0 (Trifo-AI) - MHSA 4-Heads + LayerNorm + RAG + Agente
cd /d "%~dp0"
echo Iniciando NeuroZero 3.0 (Python + NumPy puro)...
python server.py --open
pause
