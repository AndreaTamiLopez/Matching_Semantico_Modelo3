from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .utils import clean_text, format_for_model
from .ollama_client import ollama_is_up, ollama_rerank


def semantic_project_merge_llm_rerank_preciso(
    df_politicas: pd.DataFrame,
    df_proyectos: pd.DataFrame,
    col_text_politica: str,
    col_text_proyecto: str,
    col_id_proyecto: str,

    # Salida
    top_k: int = 10,
    top_k_llm: int = 5,

    # Embeddings (recall alto)
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 16,
    top_n_candidates: int = 120,
    min_bi_score: float = 0.25,

    # LLM rerank (precisión)
    use_llm_rerank: bool = True,
    llm_candidates_cap: int = 60,
    ollama_url: str = "http://localhost:11434/api/generate",
    llm_model_name: str = "qwen2.5:3b-instruct",
    llm_temperature: float = 0.0,
    llm_timeout_sec: int = 180,

    # Score combinado final
    weight_llm: float = 0.65,
    weight_bi: float = 0.35,

    sleep_between_projects_sec: float = 0.0
) -> pd.DataFrame:
    """
    Por cada PROYECTO:
      1) Recupera top_n_candidates POLÍTICAS por embeddings (coseno)
      2) (Opcional) LLM re-rank elige top_k_llm
      3) Completa a top_k con embeddings
      4) final_score = w_llm*llm_score + w_bi*bi_score (si no hay llm_score => bi_score)
    """

    # --- Validaciones ---
    if col_text_politica not in df_politicas.columns:
        raise ValueError(f"df_politicas no tiene columna requerida: {col_text_politica}")
    for c in [col_text_proyecto, col_id_proyecto]:
        if c not in df_proyectos.columns:
            raise ValueError(f"df_proyectos no tiene columna requerida: {c}")

    top_k = int(top_k)
    top_k_llm = int(min(top_k_llm, top_k))
    top_n_candidates = max(top_k, int(top_n_candidates))

    if llm_candidates_cap < top_k_llm:
        llm_candidates_cap = top_k_llm

    # --- Check Ollama ---
    if use_llm_rerank and not ollama_is_up("http://localhost:11434/api/tags", timeout=5):
        print("⚠️ Ollama no está accesible. Continuo solo con embeddings (sin LLM).")
        use_llm_rerank = False

    # --- Textos limpios ---
    pol_texts_raw = df_politicas[col_text_politica].fillna("").astype(str).map(clean_text).tolist()
    proy_texts_raw = df_proyectos[col_text_proyecto].fillna("").astype(str).map(clean_text).tolist()

    # --- Device (cuda si disponible) ---
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # --- Embeddings ---
    bi = SentenceTransformer(model_name, device=device)

    pol_texts_for_emb = format_for_model(pol_texts_raw, mode="query", model_name=model_name)
    proy_texts_for_emb = format_for_model(proy_texts_raw, mode="passage", model_name=model_name)

    E_pol = bi.encode(pol_texts_for_emb, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    E_proy = bi.encode(proy_texts_for_emb, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    # --- Recuperación top-N ---
    n_neighbors = min(top_n_candidates, len(df_politicas))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(E_pol)

    distances, indices = nn.kneighbors(E_proy, return_distance=True)
    bi_scores = 1.0 - distances

    out_rows: List[Dict[str, Any]] = []

    for i in range(len(df_proyectos)):
        base = df_proyectos.iloc[i].to_dict()
        project_text = proy_texts_raw[i]

        # 1) candidatos por embeddings
        candidates: List[Tuple[int, str, float]] = []
        for r in range(n_neighbors):
            j = int(indices[i, r])
            sc = float(bi_scores[i, r])
            if sc < float(min_bi_score):
                continue
            candidates.append((j, pol_texts_raw[j], sc))

        if not candidates:
            continue

        candidates_for_llm = candidates[: min(len(candidates), int(llm_candidates_cap))]

        # 2) LLM rerank robusto (fallback)
        llm_selections: List[Dict[str, Any]] = []
        llm_failed_reason: Optional[str] = None

        if use_llm_rerank:
            try:
                llm_json = ollama_rerank(
                    politica_candidates=candidates_for_llm,
                    project_text=project_text,
                    top_k_llm=top_k_llm,
                    ollama_url=ollama_url,
                    model_name=llm_model_name,
                    temperature=llm_temperature,
                    timeout_sec=llm_timeout_sec,
                )
                llm_selections = llm_json.get("selections", []) or []
            except Exception as e:
                llm_failed_reason = f"{type(e).__name__}: {str(e)[:200]}"
                llm_selections = []

        # 3) Construcción salida: primero LLM, luego completar a top_k por embeddings
        used_policy_idx = set()
        rank = 0

        # 3a) Selecciones LLM
        if llm_selections:
            for sel in llm_selections:
                try:
                    cand_idx = int(sel["candidate_index"])
                    llm_score = float(sel["score"])
                    llm_reason = str(sel["reason"])
                except Exception:
                    continue

                if cand_idx < 0 or cand_idx >= len(candidates_for_llm):
                    continue

                pol_j, _, bi_sc = candidates_for_llm[cand_idx]
                if pol_j in used_policy_idx:
                    continue

                used_policy_idx.add(pol_j)
                rank += 1

                final_score = weight_llm * llm_score + weight_bi * float(bi_sc)

                out_rows.append({
                    **base,
                    "matched_politica_text": df_politicas.iloc[pol_j][col_text_politica],
                    "bi_similarity_score": float(bi_sc),
                    "llm_score": float(llm_score),
                    "llm_reason": llm_reason,
                    "final_score": float(final_score),
                    "rank": rank,
                    "device_used": device,
                    "llm_failed": False
                })

                if rank >= top_k:
                    break

        # 3b) Completar con embeddings
        if rank < top_k:
            fallback = sorted(candidates, key=lambda x: x[2], reverse=True)
            for (pol_j, _, bi_sc) in fallback:
                if rank >= top_k:
                    break
                if pol_j in used_policy_idx:
                    continue

                used_policy_idx.add(pol_j)
                rank += 1

                out_rows.append({
                    **base,
                    "matched_politica_text": df_politicas.iloc[pol_j][col_text_politica],
                    "bi_similarity_score": float(bi_sc),
                    "llm_score": np.nan,
                    "llm_reason": "fallback_by_embeddings" if llm_failed_reason is None else f"fallback_by_embeddings (llm_failed: {llm_failed_reason})",
                    "final_score": float(bi_sc),
                    "rank": rank,
                    "device_used": device,
                    "llm_failed": llm_failed_reason is not None
                })

        time.sleep(float(sleep_between_projects_sec))

    df_out = pd.DataFrame(out_rows)
    if not df_out.empty and col_id_proyecto in df_out.columns:
        df_out = df_out.sort_values([col_id_proyecto, "rank"], ascending=[True, True]).reset_index(drop=True)
    return df_out
