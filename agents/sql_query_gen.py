from __future__ import annotations
from typing_extensions import TypedDict, Annotated, Dict, List, Literal
from pathlib import Path
import ast
from decimal import Decimal
import datetime as dt
import re
import tempfile
from typing import Any, Mapping, Optional, Sequence, Union
from uuid import uuid4

import config

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


from langchain_core.tools import tool

from langgraph.graph import END, StateGraph

import duckdb_engine 
import pandas as pd
import numpy as np
import math
from pandas.api.types import is_datetime64_any_dtype

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIVERSAL_SYSTEM_MESSAGE_LIMITED = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Rules:
- USE ONLY tables and columns that appear in the provided schema description.
- Prefer SELECT queries (read-only). Do NOT create, update, insert, or drop anything.
- If the user does not specify a desired number of rows, LIMIT results to at most {top_k}.
- Order results by a relevant column when helpful.
- When aggregating (SUM/COUNT/AVG/etc.) with GROUP BY and returning only a subset of categories (e.g., via LIMIT/TOP), append an additional row labelled 'Others' that applies the same aggregation across all remaining rows not already returned.
- Return only the SQL, with no commentary or formatting fences.

{return_condition}

Schema:
{table_info}
"""

UNIVERSAL_SYSTEM_MESSAGE = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. You can order the results by a relevant column to
return the most interesting examples in the database.

Rules:
- USE ONLY tables and columns that appear in the provided schema description.
- Prefer SELECT queries (read-only). Do NOT create, update, insert, or drop anything.
- Order results by a relevant column when helpful.
- When aggregating (SUM/COUNT/AVG/etc.) with GROUP BY and returning only a subset of categories (e.g., via LIMIT/TOP), append an additional row labelled 'Others' that applies the same aggregation across all remaining rows not already returned.
- Return only the SQL, with no commentary or formatting fences.

{return_condition}

Schema:
{table_info}
"""

USER_PROMPT = "Question: {input}"

VISUALIZATION_LABELS: Dict[str, str] = {
    "bar_chart": "Bar Chart",
    "line_chart": "Line Chart",
    "scatter_plot": "Scatter Plot",
    "pie": "Pie Chart",
}

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class AnswerOutput(TypedDict):
    """Answer to user's question."""
    answer: Annotated[str, ..., "Contains answer to user's question."]
    visual_recommendations: Annotated[
        List[Literal["bar_chart", "line_chart", "scatter_plot", "pie"]],
        ...,
        "Ordered list of visualization types chosen from the enumeration.",
    ]
    measure_columns: Annotated[
        List[str],
        ...,
        "Ordered numeric columns to visualize (e.g., metrics for Y-axis).",
    ]
    label_columns: Annotated[
        List[str],
        ...,
        "Ordered label columns; the first is used as the categorical axis, remaining values annotate the chart.",
    ]


agent_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0).with_structured_output(AnswerOutput)
llm_query_gen = ChatOpenAI(model="gpt-4.1", temperature=0.0).with_structured_output(QueryOutput)


query_prompt_template = ChatPromptTemplate(
    [("system", UNIVERSAL_SYSTEM_MESSAGE), ("user", USER_PROMPT)]
)

def _sanitize_table_name(path: Path) -> str:
    base = path.stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "table"
    if base[0].isdigit():
        base = f"t_{base}"
    return base

def _attach_csvs_as_views_sqlalchemy_engine(sqlalchemy_engine, csv_paths: Sequence[str]) -> None:
    """Expose each CSV as a SQL VIEW using DuckDB's read_csv_auto."""
    from sqlalchemy import text
    with sqlalchemy_engine.begin() as conn:
        for p in map(Path, csv_paths):
            tbl = _sanitize_table_name(p)
            conn.exec_driver_sql(f"DROP VIEW IF EXISTS {tbl}")
            conn.exec_driver_sql(
                f"""
                CREATE VIEW {tbl} AS
                SELECT * FROM read_csv_auto('{p.as_posix()}');
                """
            )

def build_sql_database_from_csvs(csv_paths: Sequence[str]) -> SQLDatabase:
    """Create a SQLDatabase backed by DuckDB, with CSVs mounted as views."""
    from sqlalchemy import create_engine
    import os, tempfile
    import duckdb_engine  # ensure the "duckdb" dialect is registered

    # Use a file-backed DB so the reflection connection sees the same catalog
    db_file = os.path.join(tempfile.gettempdir(), "csv2sql.duckdb")
    engine = create_engine(f"duckdb:///{db_file}")

    _attach_csvs_as_views_sqlalchemy_engine(engine, csv_paths)

    # 👈 include views in LangChain's introspection so get_table_info() isn't empty
    db = SQLDatabase(engine, view_support=True)
    db.name = "csv_duckdb"
    return db


def coerce_rows(
    data: Union[str, Sequence[Dict[str, Any]], None]
) -> List[Dict[str, Any]]:
    """Return result rows as a list of dicts, parsing LangChain string outputs."""
    if data is None:
        return []

    parsed_data: Union[str, Sequence[Dict[str, Any]], Dict[str, Any], List[Any]]
    if isinstance(data, str):
        serialized = data.strip()
        if not serialized:
            return []
        try:
            parsed_data = ast.literal_eval(serialized)
        except (ValueError, SyntaxError):
            safe_globals = {"__builtins__": {}}
            safe_locals = {"datetime": dt, "Decimal": Decimal}
            try:
                parsed_data = eval(serialized, safe_globals, safe_locals)  # noqa: S307
            except Exception as exc:
                raise ValueError("Unable to parse SQL result rows for visualization.") from exc
    else:
        parsed_data = data

    if isinstance(parsed_data, dict):
        return [parsed_data]
    if isinstance(parsed_data, list):
        if not parsed_data:
            return []
        if all(isinstance(row, dict) for row in parsed_data):
            return parsed_data
        raise ValueError("Visualization expects rows as dictionaries keyed by column name.")
    if isinstance(parsed_data, tuple):
        return coerce_rows(list(parsed_data))

    raise TypeError(f"Unsupported data format for visualization: {type(parsed_data)!r}")


def serialize_rows(rows: Optional[Sequence[Mapping[str, Any]]]) -> str:
    """Convert rows into a deterministic string representation."""
    if not rows:
        return "[]"
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each row must be a mapping of column names to values.")
        normalized.append(dict(row))
    return repr(normalized)


def rows_to_dataframe(rows: Optional[Sequence[Mapping[str, Any]]]) -> pd.DataFrame:
    """Convert query rows into a pandas DataFrame."""
    dict_rows = coerce_rows(rows)
    if not dict_rows:
        return pd.DataFrame()
    return pd.DataFrame(dict_rows)


def _export_dataframe(df: pd.DataFrame, excel_path: Union[str, Path]) -> Path:
    """Persist DataFrame, falling back to CSV if Excel limits would be exceeded."""
    excel_path = Path(excel_path)
    max_rows, max_cols = 1_048_576, 16_384
    if len(df.index) > max_rows or len(df.columns) > max_cols:
        csv_path = excel_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    df.to_excel(excel_path, index=False)
    return excel_path


def _normalize_numeric_token(value: Any) -> Optional[Union[int, float, Decimal, str]]:
    """Attempt to coerce stringified numbers with locale-specific formatting."""
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = (
            cleaned.replace("\u00a0", "")
            .replace("\u202f", "")
            .replace(" ", "")
        )
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "")
            cleaned = cleaned.replace(",", ".")
        elif cleaned.count(",") == 1 and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

        if cleaned.count(".") > 1:
            integer_part, fractional_part = cleaned.rsplit(".", 1)
            cleaned = integer_part.replace(".", "") + f".{fractional_part}"

        cleaned = re.sub(r"[^0-9\.\-eE]", "", cleaned)
        return cleaned or None
    return value


def _prepare_numeric_dataframe(
    df: pd.DataFrame, numeric_columns: Sequence[str]
) -> Dict[str, pd.Series]:
    """Return converted numeric columns without mutating the original."""
    converted: Dict[str, pd.Series] = {}
    for col in numeric_columns:
        series = df[col].map(_normalize_numeric_token)
        converted[col] = pd.to_numeric(series, errors="coerce")
    return converted


def _format_annotation_text(row: pd.Series, columns: Sequence[str]) -> str:
    """Concatenate non-numeric column values for bar labels."""
    parts: List[str] = []
    for col in columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        parts.append(f"{col}: {value}")
    return "\n".join(parts)


def _series_is_datetime(series: pd.Series) -> bool:
    """Heuristically determine whether a column represents datetime values."""
    if is_datetime64_any_dtype(series):
        return True
    sample = next((val for val in series if val is not None and not (isinstance(val, float) and math.isnan(val))), None)
    if sample is None:
        return False
    return isinstance(
        sample,
        (
            dt.datetime,
            dt.date,
            pd.Timestamp,
            np.datetime64,
        ),
    )


def _normalize_numeric_ranges(
    df: pd.DataFrame, numeric_columns: Sequence[str]
) -> List[str]:
    """Scale numeric series to comparable magnitudes and return updated column names."""
    if not numeric_columns:
        return []

    magnitudes: Dict[str, float] = {}
    for col in numeric_columns:
        series = df[col].dropna()
        magnitudes[col] = float(series.abs().max()) if not series.empty else 0.0

    base_magnitude = max(magnitudes.values(), default=0.0)
    if base_magnitude <= 0.0:
        return list(numeric_columns)

    rename_map: Dict[str, str] = {}
    for col in numeric_columns:
        magnitude = magnitudes[col]
        if magnitude <= 0.0:
            continue
        ratio = base_magnitude / magnitude if magnitude else 1.0
        if ratio < 10:
            continue
        power = int(math.floor(math.log10(ratio)))
        factor = 10 ** power
        df[col] = df[col] * factor
        rename_map[col] = f"{col} (×{factor})"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return [rename_map.get(col, col) for col in numeric_columns]


def _filter_numeric_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: int,
    context: str,
) -> List[str]:
    """Return the subset of candidate columns that contain numeric data."""
    if not candidates:
        raise ValueError(f"{context} requires numeric value columns.")

    value_candidates = [
        col for col in candidates if not _series_is_datetime(df[col])
    ]
    if not value_candidates:
        raise ValueError(
            f"{context} cannot use datetime columns as numeric values. "
            "Please select at least one numeric column."
        )

    converted = _prepare_numeric_dataframe(df, value_candidates)
    numeric_columns = [
        col for col in value_candidates if converted[col].notna().any()
    ]
    if len(numeric_columns) < required:
        raise ValueError(
            f"{context} requires at least {required} numeric column(s); "
            f"provided columns had no numeric data."
        )
    for col in numeric_columns:
        df[col] = converted[col]
    return numeric_columns


def write_query(question : str, db: SQLDatabase)-> str:
    """Generate SQL query to fetch information (read-only)."""
    #top_k = 50
    return_condition = (
        #f"If no explicit limit is requested, add 'LIMIT {top_k}'. "
        "Avoid selecting sensitive/internal IDs unless the user asks. "
        "Use normal Russian names for derived columnns."
        #"If you aggregate with GROUP BY and restrict output to a subset of categories, include an 'Others' row using the same aggregation across remaining rows."
    )
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            #"top_k": top_k,
            "table_info": db.get_table_info(),
            "input": question,
            "return_condition": return_condition,
        }
    )
    result = llm_query_gen.invoke(prompt)
    return result["query"]

def execute_query(query: str, db: SQLDatabase):
    """Execute SQL query."""
    try:
        rows = db.run_no_throw(query, include_columns=True)
        if isinstance(rows, str) and rows.startswith("Error:"):
            return {"result": None, "error": rows}
        return {"result": rows, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def fix_query(query: str, error: str)-> str:
    """Previous SQL failed; regenerate considering the DB error."""
    prompt = (
        f"The following SQL produced an error:\n\n{query}\n\n"
        f"Database error:\n{error}\n\n"
        "Rewrite ONLY the SQL so it executes successfully against the same schema. "
        "Keep it read-only and follow the same constraints."
    )
    new_query = llm_query_gen.invoke(prompt)["query"]
    return new_query


def generate_answer(question: str, query: str, result: list)-> dict:
    """Post-process results into a succinct answer."""
    prompt = (
        "You are given a user question, the SQL query, and the SQL result rows.\n"
        "- If results are empty, say there are no matching records.\n"
        "- Otherwise, summarize relevant information clearly and list key rows/fields the user asked for.\n"
        "- Do not include internal/technical columns unless the user asked for them.\n"
        "- Always answer in Russian, except measure_columns, label_columns and visual_recommendations.\n"
        "- Provide visualization guidance by specifying:\n"
        "  * measure_columns: ordered numeric columns suitable for metrics (y-axis values).\n"
        "  * label_columns: ordered categorical/context columns (first item will be used as x-axis; remaining entries annotate data points).\n"
        "  * visual_recommendations: choose from bar_chart, line_chart, scatter_plot, pie.\n\n"
        f'Question: {question}\n'
        f'SQL Query: {query}\n'
        f'SQL Result: {result}'
    )
    answer = agent_llm.invoke(prompt)
    return answer


def create_visualization_image(
    rows: Union[str, Sequence[Dict[str, Any]]],
    chart_type: str,
    label_columns: Sequence[str],
    measure_columns: Sequence[str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Render the requested chart and persist it as an image."""
    label_columns = list(dict.fromkeys(label_columns))
    measure_columns = list(dict.fromkeys(measure_columns))
    if not label_columns:
        raise ValueError("At least one label column must be provided for visualization.")

    #rows = _coerce_rows(data)
    #if not rows:
    #    raise ValueError("No rows available to visualize.")

    df = pd.DataFrame(rows)
    required_columns = list(dict.fromkeys(label_columns + measure_columns))
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in result set: {missing_columns}")

    chart_key = chart_type.lower()
    if chart_key not in VISUALIZATION_LABELS:
        raise ValueError(f"Unsupported visualization type: {chart_type}")

    df_plot = df[required_columns].copy()
    dimension_column = label_columns[0]
    annotation_columns = label_columns[1:]

    numeric_candidates: List[str] = []
    numeric_columns: List[str] = []
    if chart_key in {"bar_chart", "line_chart"}:
        if not measure_columns:
            raise ValueError(f"{chart_type} requires at least one numeric column.")
        numeric_candidates = list(measure_columns)
        numeric_columns = _filter_numeric_columns(
            df_plot, numeric_candidates, required=1, context=chart_type
        )
        if len(numeric_columns) > 1:
            numeric_columns = _normalize_numeric_ranges(df_plot, numeric_columns)
    elif chart_key == "scatter_plot":
        if len(measure_columns) < 2:
            extra_candidates = [
                col
                for col in df_plot.columns
                if col not in label_columns and col not in measure_columns
            ]
            measure_columns = list(measure_columns) + extra_candidates
        if len(measure_columns) < 2:
            raise ValueError("scatter_plot requires two numeric columns.")
        numeric_candidates = list(measure_columns[:2])
        numeric_columns = _filter_numeric_columns(
            df_plot, numeric_candidates, required=2, context=chart_type
        )[:2]
    elif chart_key == "pie":
        if not measure_columns:
            raise ValueError("pie requires a numeric value column.")
        numeric_candidates = [measure_columns[0]]
        numeric_columns = _filter_numeric_columns(
            df_plot, numeric_candidates, required=1, context=chart_type
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    title = VISUALIZATION_LABELS[chart_key]
    if chart_key == "bar_chart":
        y_cols = numeric_columns
        df_plot.plot(
            kind="bar",
            x=dimension_column,
            y=y_cols[0] if len(y_cols) == 1 else y_cols,
            ax=ax,
        )
        ax.set_xlabel(dimension_column)
        ax.set_ylabel(", ".join(y_cols))
        if len(y_cols) == 1 and annotation_columns:
            y_min, y_max = ax.get_ylim()
            if y_max > y_min:
                padding = (y_max - y_min) * 0.15
                ax.set_ylim(y_min, y_max + padding)
            annotations = [
                _format_annotation_text(row, annotation_columns)
                for _, row in df_plot.iterrows()
            ]
            for patch, text in zip(ax.patches, annotations):
                if not text:
                    continue
                height = patch.get_height()
                ax.annotate(
                    text,
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 6),
                    textcoords="offset points",
                    annotation_clip=False,
                )
    elif chart_key == "line_chart":
        y_cols = numeric_columns
        df_plot.plot(
            kind="line",
            marker="o",
            x=dimension_column,
            y=y_cols[0] if len(y_cols) == 1 else y_cols,
            ax=ax,
        )
        ax.set_xlabel(dimension_column)
        ax.set_ylabel(", ".join(y_cols))
    elif chart_key == "scatter_plot":
        x_col, y_col = numeric_columns[:2]
        ax.scatter(df_plot[x_col], df_plot[y_col], color="#1f77b4", alpha=0.85)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    else:  # pie
        label_col = dimension_column
        value_col = numeric_columns[0]
        pie_series = df_plot.groupby(label_col, dropna=False)[value_col].sum()
        if pie_series.empty:
            raise ValueError("No values available to render a pie chart.")
        ax.pie(
            pie_series.values,
            labels=pie_series.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_ylabel("")

    ax.set_title(title)
    if chart_key in {"bar_chart", "line_chart"} and len(numeric_columns) > 1:
        ax.legend(loc="best")

    if output_path is None:
        output_path = Path(f"./images/visualization_{uuid4().hex}.png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def get_response(question: str, data_paths: List[str])-> dict:
    db = build_sql_database_from_csvs(["data/data.csv"])
    query = write_query(question, db)
    for attempt in range(3):
        result = execute_query(query, db)
        if result.get("error") is None:
            break
        if attempt == 2:
            raise Exception(result["error"])
        query = fix_query(query, result["error"])
    
    top_k = 30
    db_result = result.get("result", [])
    rows = coerce_rows(db_result)
    df = rows_to_dataframe(rows)
    export_path = _export_dataframe(df, f"data/data_{uuid4().hex}.xlsx")
    answer = generate_answer(question, query, serialize_rows(rows[:top_k]))

    viz_methods = answer.get("visual_recommendations") or []
    measure_columns = answer.get("measure_columns") or []
    label_columns = answer.get("label_columns") or []
    image_path = ""
    if viz_methods and measure_columns and label_columns:
        try:
            image_path = create_visualization_image(
                rows,
                viz_methods[0],
                label_columns,
                measure_columns,
            )
            print(f"Visualization image saved to: {image_path}")
        except Exception as viz_error:
            print(f"Visualization generation failed: {viz_error}")
    return {
        "query": query,
        "answer": answer.get('answer', ''),
        "data": export_path,
        "image": image_path,
        "graph_type": answer.get('visual_recommendations', '')
    }

if __name__ == "__main__": 
    #question = "Верни самые популярные страны."
    #question = "Какая доля продаж приходится на каждый регион за последний квартал?"
    #question = "Как связаны сумма страховой выплаты и длительность поездки по всем договорам за 2024 год?"
    #question = "Как менялась суммарная страховая премия по месяцу в 2024 году"
    question = "Какие самые крупные выплаты были?"
    db = build_sql_database_from_csvs(["data/data.csv"])
    query = write_query(question, db)
    for _ in range(3):
        result = execute_query(query, db)
        if result.get("error") is None:
            break
        query = fix_query(query, result["error"])
    db_result = result.get("result", [])
    rows = coerce_rows(db_result)
    df = rows_to_dataframe(rows)
    export_path = _export_dataframe(df, "data/gen_data.xlsx")
    print(f"Tabular data saved to: {export_path}")
    k_top = 30
    answer = generate_answer(question, query, serialize_rows(rows[:k_top]))
    print(f"Question: {question}")
    print(f"Query: {query}")
    print(f"Data: {db_result}")
    print(f"Answer: {answer.get('answer', '')}")
    print(f"Visualization: {answer.get('visual_recommendations', '')}")
    print(f"Measures: {answer.get('measure_columns', '')}")
    print(f"Labels: {answer.get('label_columns', '')}")
    viz_methods = answer.get("visual_recommendations") or []
    measure_columns = answer.get("measure_columns") or []
    label_columns = answer.get("label_columns") or []
    if viz_methods and measure_columns and label_columns:
        try:
            image_path = create_visualization_image(
                rows,
                viz_methods[0],
                label_columns,
                measure_columns,
            )
            print(f"Visualization image saved to: {image_path}")
        except Exception as viz_error:
            print(f"Visualization generation failed: {viz_error}")
