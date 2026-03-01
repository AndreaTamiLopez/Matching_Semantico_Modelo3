import re
import json
from typing import Any, Dict, List


def clean_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_for_model(texts: List[str], mode: str, model_name: str) -> List[str]:
    """
    Para modelos tipo E5/BGE, usar prefijos mejora la calidad:
      - query: (consulta) -> política
      - passage: (documento) -> proyecto
    """
    ml = (model_name or "").lower()
    if ("e5" in ml) or ("bge" in ml):
        prefix = "query: " if mode == "query" else "passage: "
        return [prefix + t for t in texts]
    return texts


def extract_json_loose(text: str) -> Dict[str, Any]:
    """
    Extrae JSON aunque el modelo meta texto extra.
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No se encontró un bloque JSON en la respuesta del LLM.")
    return json.loads(m.group(0).strip())


def validate_llm_schema(obj: Dict[str, Any]) -> None:
    """
    Espera:
      { "selections": [ {"candidate_index": int, "score": float 0-1, "reason": str} ] }
    """
    if "selections" not in obj or not isinstance(obj["selections"], list):
        raise ValueError("LLM: falta 'selections' o no es lista.")
    if len(obj["selections"]) == 0:
        raise ValueError("LLM: 'selections' viene vacía.")

    for it in obj["selections"]:
        if not isinstance(it, dict):
            raise ValueError("LLM: selection no es dict.")
        for k in ["candidate_index", "score", "reason"]:
            if k not in it:
                raise ValueError(f"LLM: falta campo {k}.")
        sc = float(it["score"])
        if sc < 0 or sc > 1:
            raise ValueError("LLM: score fuera de [0,1].")
