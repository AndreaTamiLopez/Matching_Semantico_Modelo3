# Matching semántico con LLM Re-Ranking (Ollama)

Este repositorio implementa un sistema de emparejamiento semántico entre:
- **Indicadores de proyectos (PATR)**, y
- **Catálogo de políticas/indicadores (MGA / SisPT)**

El objetivo es obtener, para cada proyecto, el **Top-K** de políticas más relevantes.

---

## Enfoque: dos etapas + fallback robusto

El pipeline combina precisión y escalabilidad usando tres componentes:

1) **Recuperación (recall alto) con embeddings**
- Se generan embeddings con Sentence Transformers (por defecto `BAAI/bge-m3`).
- Se recuperan **Top-N** candidatas por similitud coseno (kNN).
- Se aplica un umbral mínimo `min_bi_score` para evitar ruido.

2) **Re-ranking con LLM vía Ollama (opcional)**
- Un LLM local (por ejemplo `qwen2.5:3b-instruct` o `deepseek-r1:7b`) revisa un subconjunto de candidatas.
- Devuelve un JSON con selecciones, score 0–1 y una razón corta.
- Si Ollama no está disponible o el LLM falla, el sistema continúa con embeddings (fallback).

3) **Score final combinado**
- Si existe salida del LLM:
  - `final_score = weight_llm * llm_score + weight_bi * bi_similarity_score`
- Si no existe LLM:
  - `final_score = bi_similarity_score`

---

## Requisitos

- Python 3.9+
- (Opcional) Ollama corriendo localmente en `http://localhost:11434`

Dependencias: ver `requirements.txt`.

---

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
