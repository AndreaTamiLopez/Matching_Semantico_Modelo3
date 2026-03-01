import os
import pandas as pd

from semantic_matching_ollama import (
    enable_hf_ssl_fix,
    semantic_project_merge_llm_rerank_preciso,
    export_matches_to_excel
)


def main():
    # (Opcional) Fix SSL si HuggingFace falla al descargar modelos
    # enable_hf_ssl_fix()

    # Inputs locales
    politicas_path = "data/raw/politicas.xlsx"
    proyectos_path = "data/raw/proyectos.xlsx"

    df_politicas = pd.read_excel(politicas_path)
    df_proyectos = pd.read_excel(proyectos_path)

    df_out = semantic_project_merge_llm_rerank_preciso(
        df_politicas=df_politicas,
        df_proyectos=df_proyectos,
        col_text_politica="Indicador de Producto(MGA)",
        col_text_proyecto="Indicadores de producto PATR",
        col_id_proyecto="codigo_proyecto",

        top_k=10,
        top_k_llm=5,

        model_name="BAAI/bge-m3",
        batch_size=16,
        top_n_candidates=120,
        min_bi_score=0.25,

        use_llm_rerank=True,
        llm_model_name="qwen2.5:3b-instruct",
        llm_candidates_cap=15,
        llm_timeout_sec=180,
        llm_temperature=0.0,
    )

    os.makedirs("data/outputs", exist_ok=True)
    out_path = "data/outputs/matching_ollama_topk.xlsx"

    export_matches_to_excel(
        df_out=df_out,
        path_xlsx=out_path,
        project_id_col="codigo_proyecto",
        make_top1_sheet=True
    )

    print("Excel guardado en:", out_path)


if __name__ == "__main__":
    main()
