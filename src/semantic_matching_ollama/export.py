import pandas as pd


def export_matches_to_excel(
    df_out: pd.DataFrame,
    path_xlsx: str,
    project_id_col: str,
    make_top1_sheet: bool = True
) -> None:
    """
    Exporta a Excel:
      - Hoja 1: top_k_por_proyecto (completo)
      - Hoja 2: top_1_por_proyecto (opcional)
    """
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name="top_k_por_proyecto", index=False)

        if make_top1_sheet and (project_id_col in df_out.columns):
            df_top1 = (
                df_out
                .sort_values([project_id_col, "rank"])
                .groupby(project_id_col, as_index=False)
                .first()
            )
            df_top1.to_excel(writer, sheet_name="top_1_por_proyecto", index=False)
