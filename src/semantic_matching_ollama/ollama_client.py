import time
import requests
from typing import Any, Dict, List, Optional, Tuple

from .utils import extract_json_loose, validate_llm_schema


def ollama_is_up(url: str = "http://localhost:11434/api/tags", timeout: int = 5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ollama_rerank(
    politica_candidates: List[Tuple[int, str, float]],
    project_text: str,
    top_k_llm: int,
    ollama_url: str = "http://localhost:11434/api/generate",
    model_name: str = "deepseek-r1:7b",
    temperature: float = 0.0,
    timeout_sec: int = 600,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    politica_candidates: lista de (policy_index_en_df, policy_text, bi_score)

    Devuelve JSON:
      { "selections": [ {"candidate_index": i, "score": 0-1, "reason": "..."} ] }

    candidate_index se refiere al índice dentro de politica_candidates (0..len-1)
    """
    candidates_txt = "\n".join([
        f"{i}. (bi_score={bi_score:.3f}) {pol_txt}"
        for i, (_, pol_txt, bi_score) in enumerate(politica_candidates)
    ])

    prompt = f"""
Eres un experto en políticas públicas y formulación de proyectos.

TAREA:
Selecciona las {top_k_llm} POLÍTICAS que mejor correspondan al PROYECTO.

PROYECTO:
\"\"\"{project_text}\"\"\"

POLÍTICAS CANDIDATAS:
{candidates_txt}

CRITERIOS (prioridad):
1) Misma finalidad/objetivo (qué problema público atiende)
2) Misma población objetivo / beneficiarios
3) Misma intervención o instrumento (infraestructura, subsidio, regulación, fortalecimiento institucional, tecnología, etc.)
4) Mismo sector/tema (salud, educación, agua, vivienda, empleo, ambiente, seguridad, etc.)
5) Nivel de especificidad: prefiere la política más específica y directamente aplicable

SALIDA:
Devuelve SOLO JSON (sin texto adicional) con el formato:

{{
  "selections": [
    {{"candidate_index": 0, "score": 0.0, "reason": "1 frase"}}
  ]
}}

REGLAS:
- score entre 0 y 1.
- reason máximo 1 frase.
- candidate_index debe ser un índice del listado de candidatas.
""".strip()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(ollama_url, json=payload, timeout=timeout_sec)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")

            obj = extract_json_loose(r.json().get("response", ""))
            validate_llm_schema(obj)
            return obj

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"LLM rerank falló tras {max_retries} intentos. Último error: {last_err}")
