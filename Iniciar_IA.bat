@echo off
chcp 65001 >nul
title NeuroZero 2.0 - Micro-Transformer + RAG + Agente
cd /d "%~dp0"
echo Iniciando NeuroZero 2.0 (Python + NumPy puro)...
python server.py --open
pause
