import os
import re
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
from openai import OpenAI

try:
    from rag_engine import (
        ingest_documents,
        search_knowledge,
        clear_vector_db,
        get_indexed_documents_count,
        format_retrieved_context,
        KNOWLEDGE_BASE_DIR,
        CHROMA_DB_DIR,
    )
    RAG_IMPORT_ERROR = None
except Exception as rag_import_exception:
    RAG_IMPORT_ERROR = rag_import_exception

try:
    from agents import (
        AGENT_DEFINITIONS,
        AGENT_HISTORY_DIR,
        AGENT_REPORTS_DIR,
        AGENT_TEAM_ORDER,
        build_agent_prompt,
        build_agent_team_synthesis_prompt,
        save_agent_report,
        save_agent_team_report,
    )
    AGENTS_IMPORT_ERROR = None
except Exception as agents_import_exception:
    AGENTS_IMPORT_ERROR = agents_import_exception

try:
    from memory_engine import (
        DATABASE_CONTEXT_JSON_PATH,
        DATABASE_CONTEXT_MD_PATH,
        MEMORY_DIR,
        TASK_HISTORY_PATH,
        format_database_context_for_prompt,
        get_memory_stats,
        load_agent_task_history,
        save_agent_task_memory,
        scan_sql_server_context,
        search_database_context,
        summarize_sql_execution_records,
    )
    MEMORY_IMPORT_ERROR = None
except Exception as memory_import_exception:
    MEMORY_IMPORT_ERROR = memory_import_exception

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

st.set_page_config(
    page_title="Local SQL Analytics Workspace",
    layout="wide"
)

st.title("Local SQL Analytics Workspace")
st.caption("Explore SQL Server data, run local AI analysis, and prepare BI outputs.")

st.info(
    "Choose a database and table from the sidebar, inspect the data, then use Qwen or the agents "
    "when you need SQL help, explanations, checks, or dashboard ideas. Everything runs locally and SQL is read-only."
)

with st.expander("Core tools in this workspace", expanded=False):
    st.markdown(
        """
- **SQL Server** is the source of the data.
- **pandas** handles previews, checks, KPIs, charts, and exports after data is loaded.
- **Local Qwen** helps explain tables, draft SQL, and summarize query results.
- **RAG / ChromaDB** adds context from local documents in `knowledge_base/`.
- **AI agents** split the work into profiling, SQL, modeling, anomaly checks, BI planning, and validation.
- **Power BI outputs** are local files you can open later in Power BI Desktop.
"""
    )

# -----------------------------
# SETTINGS
# -----------------------------

SERVER = "localhost"
LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "qwen/qwen3.5-9b"
AUTO_QWEN_MAX_TOKENS = 16000
AUTO_PROMPT_CHAR_LIMIT = 45000
LMSTUDIO_TIMEOUT_SECONDS = 1800
BROWSER_PREVIEW_ROWS = 2000
QUERY_RESULT_PREVIEW_ROWS = 5000
MAX_CHART_POINTS = 5000
AGENT_SQL_MAX_QUERIES = 3
AGENT_SQL_RESULT_LIMIT = 5000
POWERBI_OUTPUT_DIR = os.path.join("outputs", "powerbi")
SHOW_ADVANCED_BI_TRAINING = False


# -----------------------------
# SQL CONNECTION
# -----------------------------

def connect_to_database(database_name="master"):
    connection_string = (
        "DRIVER={SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={database_name};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(connection_string)


def get_databases():
    conn = connect_to_database("master")
    query = """
    SELECT name
    FROM sys.databases
    WHERE database_id > 4
    ORDER BY name
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df["name"].tolist()


def get_tables(database_name):
    conn = connect_to_database(database_name)
    query = """
    SELECT TABLE_SCHEMA, TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["FULL_TABLE_NAME"] = df["TABLE_SCHEMA"] + "." + df["TABLE_NAME"]
    return df


def get_tables_for_databases(database_names):
    # Load table lists from every selected database.
    # Streamlit can then show one combined table dropdown.
    all_tables = []
    failed_databases = []

    for database_name in database_names:
        try:
            database_tables = get_tables(database_name)
            database_tables.insert(0, "DATABASE_NAME", database_name)
            database_tables["DISPLAY_TABLE_NAME"] = (
                database_name + "." + database_tables["FULL_TABLE_NAME"]
            )
            all_tables.append(database_tables)
        except Exception as e:
            failed_databases.append((database_name, str(e)))

    if not all_tables:
        empty_tables = pd.DataFrame(
            columns=[
                "DATABASE_NAME",
                "TABLE_SCHEMA",
                "TABLE_NAME",
                "FULL_TABLE_NAME",
                "DISPLAY_TABLE_NAME",
            ]
        )
        return empty_tables, failed_databases

    return pd.concat(all_tables, ignore_index=True), failed_databases


def load_table(database_name, full_table_name, top_n):
    conn = connect_to_database(database_name)

    schema_name, table_name = full_table_name.split(".", 1)

    query = f"""
    SELECT TOP {top_n} *
    FROM [{schema_name}].[{table_name}]
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_selected_table_targets(table_targets_df, top_n):
    # A target is one database + one schema.table.
    # When several targets are loaded, pandas combines them and adds source columns.
    loaded_frames = []
    load_errors = []
    add_source_columns = len(table_targets_df) > 1

    for _, target in table_targets_df.iterrows():
        database_name = target["DATABASE_NAME"]
        full_table_name = target["FULL_TABLE_NAME"]
        display_name = target["DISPLAY_TABLE_NAME"]

        try:
            target_df = load_table(database_name, full_table_name, top_n)

            if add_source_columns:
                target_df.insert(0, "SourceTable", full_table_name)
                target_df.insert(0, "SourceDatabase", database_name)

            loaded_frames.append(target_df)
        except Exception as e:
            load_errors.append((display_name, str(e)))

    if not loaded_frames:
        return pd.DataFrame(), load_errors

    combined_df = pd.concat(loaded_frames, ignore_index=True, sort=False)
    return combined_df, load_errors


def strip_sql_comments(sql_query):
    sql_query = re.sub(r"/\*.*?\*/", " ", sql_query, flags=re.DOTALL)
    sql_query = re.sub(r"--.*?$", " ", sql_query, flags=re.MULTILINE)
    return sql_query


def validate_read_only_sql(sql_query):
    cleaned_query = strip_sql_comments(sql_query).strip()
    normalized_query = re.sub(r"\s+", " ", cleaned_query).lower()

    if not normalized_query:
        return False, "The SQL query is empty."

    if not (normalized_query.startswith("select ") or normalized_query.startswith("with ")):
        return False, "Only SELECT queries are allowed."

    if ";" in cleaned_query.rstrip(";"):
        return False, "Only one SQL statement is allowed."

    blocked_patterns = [
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\bcreate\b",
        r"\btruncate\b",
        r"\bmerge\b",
        r"\bexec\b",
        r"\bexecute\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\bbackup\b",
        r"\brestore\b",
        r"\bdbcc\b",
        r"\buse\b",
        r"\binto\b",
        r"\bxp_\w+",
        r"\bsp_\w+",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, normalized_query):
            return False, "This query contains a blocked keyword or command."

    return True, "Query is read-only."


def clean_sql_candidate(candidate_text):
    candidate_lines = []
    started_sql = False
    heading_pattern = re.compile(
        r"^(explanation|purpose|ordering|limitation|note|notes|tables|table|columns|"
        r"result|results|why|because|this query|extra context|data quality|kpis|"
        r"dashboard|relationships)\b",
        flags=re.IGNORECASE,
    )

    for line in candidate_text.splitlines():
        stripped_line = line.strip()
        lower_line = stripped_line.lower()

        if "```" in stripped_line:
            if started_sql:
                break
            continue

        if not started_sql:
            if re.match(r"^(select|with)\b", stripped_line, flags=re.IGNORECASE):
                started_sql = True
            else:
                continue

        if not stripped_line:
            continue

        if candidate_lines and (
            heading_pattern.match(lower_line.rstrip(":"))
            or lower_line.startswith(("- ", "* ", "#"))
            or re.match(r"^[A-Za-z][A-Za-z ]+:$", stripped_line)
        ):
            break

        if ";" in line:
            before_semicolon = line.split(";", 1)[0].rstrip()
            if before_semicolon:
                candidate_lines.append(before_semicolon)
            break

        candidate_lines.append(line.rstrip())

    extracted_query = "\n".join(candidate_lines).strip().strip("`").rstrip(";").strip()
    return extracted_query or None


def extract_sql_query(answer):
    # Prefer fenced code blocks, but clean them because local LLMs sometimes put labels
    # like "Tables:" or "Explanation:" inside the same block as the SQL.
    fenced_matches = re.findall(r"```(?:sql)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    for fenced_text in fenced_matches:
        sql_query = clean_sql_candidate(fenced_text)
        if sql_query:
            return sql_query

    sql_start = re.search(r"\b(select|with)\b", answer, flags=re.IGNORECASE)
    if not sql_start:
        return None

    return clean_sql_candidate(answer[sql_start.start():])


def extract_all_sql_queries(answer):
    # Agent reports can contain more than one SQL draft. This helper extracts each
    # fenced SQL block so the app can save and safety-check them one by one.
    sql_queries = []
    fenced_matches = re.findall(r"```(?:sql)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)

    for fenced_text in fenced_matches:
        sql_query = clean_sql_candidate(fenced_text)
        if sql_query and sql_query not in sql_queries:
            sql_queries.append(sql_query)

    if not sql_queries:
        sql_query = extract_sql_query(answer)
        if sql_query:
            sql_queries.append(sql_query)

    return sql_queries


def format_sql_drafts(sql_queries):
    if not sql_queries:
        return None

    formatted_queries = []
    for index, sql_query in enumerate(sql_queries, start=1):
        formatted_queries.append(f"-- Query {index}\n{sql_query.strip().rstrip(';')};")

    return "\n\n".join(formatted_queries)


def build_sql_safety_summary(sql_queries):
    safety_rows = []

    for index, sql_query in enumerate(sql_queries, start=1):
        is_safe, safety_message = validate_read_only_sql(sql_query)
        safety_rows.append({
            "Query": index,
            "ReadOnlySafe": bool(is_safe),
            "Status": safety_message,
            "Preview": sql_query.strip().replace("\n", " ")[:180],
        })

    return safety_rows


def normalize_sql_server_dialect(sql_query):
    # Local LLMs sometimes write PostgreSQL syntax. Convert common cases to SQL Server T-SQL.
    normalized_query = sql_query

    normalized_query = re.sub(
        r"DATE_TRUNC\s*\(\s*'month'\s*,\s*([A-Za-z_][\w.\[\]]*)\s*\)",
        r"DATEFROMPARTS(YEAR(\1), MONTH(\1), 1)",
        normalized_query,
        flags=re.IGNORECASE,
    )
    normalized_query = re.sub(
        r"DATE_TRUNC\s*\(\s*'year'\s*,\s*([A-Za-z_][\w.\[\]]*)\s*\)",
        r"DATEFROMPARTS(YEAR(\1), 1, 1)",
        normalized_query,
        flags=re.IGNORECASE,
    )
    normalized_query = re.sub(
        r"DATE_TRUNC\s*\(\s*'day'\s*,\s*([A-Za-z_][\w.\[\]]*)\s*\)",
        r"CAST(\1 AS date)",
        normalized_query,
        flags=re.IGNORECASE,
    )

    return normalized_query


def bracket_full_table_name(full_table_name):
    schema_name, table_name = full_table_name.split(".", 1)
    return f"[{schema_name}].[{table_name}]"


def quote_sql_identifier(identifier):
    return "[" + str(identifier).replace("]", "]]") + "]"


def safe_preview_text(value, max_length):
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def get_column_lookup(dtype_df):
    return {column.lower(): column for column in dtype_df["Column"].tolist()}


def show_dataframe_preview(dataframe, max_rows, context_label):
    # Sending hundreds of thousands of rows to the browser can exceed Streamlit's
    # message limit. Keep the full dataframe in Python, but render only a preview.
    total_rows = len(dataframe)
    preview_df = dataframe.head(max_rows)

    st.dataframe(preview_df, width="stretch")

    if total_rows > max_rows:
        st.caption(
            f"Showing first {max_rows:,} of {total_rows:,} {context_label} rows. "
            "Calculations still use the loaded data in Python; the preview is capped "
            "to keep the browser responsive."
        )


def build_revenue_trend_sql(user_question, selected_table, dtype_df):
    question = user_question.lower()
    columns = get_column_lookup(dtype_df)

    trend_words = ["trend", "over time", "line chart", "monthly", "month", "revenue"]
    wants_trend = any(word in question for word in trend_words)

    if not wants_trend:
        return None

    if "paymentdate" not in columns or "paymentmoney" not in columns:
        return None

    payment_date_col = columns["paymentdate"]
    payment_money_col = columns["paymentmoney"]
    table_name = bracket_full_table_name(selected_table)

    return f"""
SELECT
    DATEFROMPARTS(YEAR([{payment_date_col}]), MONTH([{payment_date_col}]), 1) AS PaymentMonth,
    SUM(CAST([{payment_money_col}] AS float)) AS TotalPayments
FROM {table_name}
WHERE [{payment_date_col}] IS NOT NULL
  AND [{payment_money_col}] IS NOT NULL
GROUP BY DATEFROMPARTS(YEAR([{payment_date_col}]), MONTH([{payment_date_col}]), 1)
ORDER BY PaymentMonth
""".strip()


def build_time_aggregate_sql(selected_table, date_column, value_column, aggregation, time_grain):
    table_name = bracket_full_table_name(selected_table)
    date_identifier = quote_sql_identifier(date_column)

    if time_grain == "Day":
        date_expression = f"CAST({date_identifier} AS date)"
        period_alias = "PeriodDay"
    elif time_grain == "Year":
        date_expression = f"DATEFROMPARTS(YEAR({date_identifier}), 1, 1)"
        period_alias = "PeriodYear"
    else:
        date_expression = f"DATEFROMPARTS(YEAR({date_identifier}), MONTH({date_identifier}), 1)"
        period_alias = "PeriodMonth"

    if aggregation == "COUNT rows":
        value_expression = "COUNT(*)"
        value_alias = "RowCount"
        where_clause = f"WHERE {date_identifier} IS NOT NULL"
    else:
        value_identifier = quote_sql_identifier(value_column)
        value_expression = f"{aggregation}(CAST({value_identifier} AS float))"
        value_alias = f"{aggregation}_{value_column}"
        where_clause = f"WHERE {date_identifier} IS NOT NULL AND {value_identifier} IS NOT NULL"

    return f"""
SELECT
    {date_expression} AS {period_alias},
    {value_expression} AS {quote_sql_identifier(value_alias)}
FROM {table_name}
{where_clause}
GROUP BY {date_expression}
ORDER BY {period_alias}
""".strip()


def get_full_chart_column_candidates(df):
    # These candidates come from the loaded preview rows, but the final chart query
    # runs against the full SQL table in SQL Server.
    date_candidates = []
    numeric_candidates = []

    for column in df.columns:
        sample = df[column].dropna().head(100)
        if sample.empty:
            continue

        column_name = str(column).lower()

        if (
            "date" in column_name
            or "time" in column_name
            or "month" in column_name
            or pd.api.types.is_datetime64_any_dtype(df[column])
        ):
            parsed_dates = pd.to_datetime(sample, errors="coerce")
            if parsed_dates.notna().any():
                date_candidates.append(column)

        numeric_values = pd.to_numeric(sample, errors="coerce")
        required_matches = max(1, int(len(sample) * 0.8))
        if numeric_values.notna().sum() >= required_matches:
            numeric_candidates.append(column)

    return date_candidates, numeric_candidates


def run_read_only_query(database_name, sql_query):
    sql_query = normalize_sql_server_dialect(sql_query)
    is_safe, safety_message = validate_read_only_sql(sql_query)
    if not is_safe:
        raise ValueError(safety_message)

    conn = connect_to_database(database_name)
    try:
        result_df = pd.read_sql(sql_query, conn)
    finally:
        conn.close()

    return result_df


def run_read_only_query_for_table_targets(table_targets, sql_query):
    # Multi-database SQL mode:
    # Run the same read-only SELECT inside each target database, then combine results.
    # This works best when every target has the same schema.table, for example dbo.Payment.
    sql_query = normalize_sql_server_dialect(sql_query)
    is_safe, safety_message = validate_read_only_sql(sql_query)
    if not is_safe:
        raise ValueError(safety_message)

    result_frames = []
    target_errors = []

    for target in table_targets:
        database_name = target["DATABASE_NAME"]
        full_table_name = target["FULL_TABLE_NAME"]
        display_name = target["DISPLAY_TABLE_NAME"]
        conn = None

        try:
            conn = connect_to_database(database_name)
            target_result = pd.read_sql(sql_query, conn)
            target_result.insert(0, "SourceTable", full_table_name)
            target_result.insert(0, "SourceDatabase", database_name)
            result_frames.append(target_result)
        except Exception as e:
            target_errors.append((display_name, str(e)))
        finally:
            if conn is not None:
                conn.close()

    if not result_frames:
        error_text = "\n".join(
            f"{display_name}: {error_text}"
            for display_name, error_text in target_errors
        )
        raise ValueError(f"No selected database returned a SQL result.\n{error_text}")

    combined_result = pd.concat(result_frames, ignore_index=True, sort=False)
    return combined_result, target_errors


def show_auto_result_chart(result_df):
    if result_df is None or result_df.empty:
        return

    date_column = None
    numeric_column = None

    for column in result_df.columns:
        column_name = str(column).lower()
        if "date" in column_name or "month" in column_name or "time" in column_name:
            parsed_dates = pd.to_datetime(result_df[column], errors="coerce")
            if parsed_dates.notna().any():
                date_column = column
                break

    numeric_columns = result_df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        numeric_column = numeric_columns[0]

    if date_column is None or numeric_column is None:
        return

    group_column = None
    for possible_group in ["SourceDatabase", "SourceTable"]:
        if possible_group in result_df.columns:
            group_column = possible_group
            break

    chart_columns = [date_column, numeric_column]
    if group_column:
        chart_columns.append(group_column)

    chart_df = result_df[chart_columns].copy()
    chart_df[date_column] = pd.to_datetime(chart_df[date_column], errors="coerce")
    chart_df[numeric_column] = pd.to_numeric(chart_df[numeric_column], errors="coerce")
    chart_df = chart_df.dropna().sort_values(date_column)

    if chart_df.empty:
        return

    if len(chart_df) > MAX_CHART_POINTS:
        st.caption(
            f"Chart preview uses first {MAX_CHART_POINTS:,} of {len(chart_df):,} points. "
            "For large tables, prefer grouped SQL charts such as monthly totals."
        )
        chart_df = chart_df.head(MAX_CHART_POINTS)

    st.markdown("### Auto Chart")

    if group_column:
        grouped_chart_df = chart_df.pivot_table(
            index=date_column,
            columns=group_column,
            values=numeric_column,
            aggfunc="sum",
        ).sort_index()
        st.line_chart(grouped_chart_df)
    else:
        st.line_chart(chart_df.set_index(date_column)[numeric_column])


def run_read_only_query_for_current_selection(
    sql_query,
    selected_database,
    selected_table_targets,
    multi_database_sql_enabled,
):
    if multi_database_sql_enabled:
        return run_read_only_query_for_table_targets(
            selected_table_targets,
            sql_query,
        )

    return run_read_only_query(selected_database, sql_query), []


def add_top_limit_to_select(sql_query, row_limit):
    # Agent SQL should answer questions, not stream hundreds of thousands of raw rows.
    # For simple SELECT queries without TOP/OFFSET/FETCH, insert TOP as a safety cap.
    stripped_query = sql_query.strip().rstrip(";")
    lowered_query = re.sub(r"\s+", " ", stripped_query).lower()

    if not lowered_query.startswith("select "):
        return stripped_query

    if re.search(r"\btop\s+\(?\d+\)?\b|\boffset\b|\bfetch\b", lowered_query):
        return stripped_query

    if re.match(r"^\s*select\s+distinct\s+", stripped_query, flags=re.IGNORECASE):
        return re.sub(
            r"^\s*select\s+distinct\s+",
            f"SELECT DISTINCT TOP {row_limit} ",
            stripped_query,
            count=1,
            flags=re.IGNORECASE,
        )

    return re.sub(
        r"^\s*select\s+",
        f"SELECT TOP {row_limit} ",
        stripped_query,
        count=1,
        flags=re.IGNORECASE,
    )


def execute_agent_sql_queries(
    sql_queries,
    sql_execution_enabled,
    multi_database_sql_enabled,
    selected_database,
    selected_table_targets,
):
    execution_records = []

    for index, sql_query in enumerate(sql_queries[:AGENT_SQL_MAX_QUERIES], start=1):
        normalized_sql = normalize_sql_server_dialect(sql_query)
        limited_sql = add_top_limit_to_select(normalized_sql, AGENT_SQL_RESULT_LIMIT)
        is_safe, safety_message = validate_read_only_sql(limited_sql)

        record = {
            "query_index": index,
            "sql": limited_sql,
            "read_only_safe": bool(is_safe),
            "status": safety_message,
            "result": None,
            "target_errors": [],
            "error": None,
        }

        if not sql_execution_enabled:
            record["status"] = (
                "SQL was not executed because this selection has multiple different table targets. "
                "Use one exact table or the same schema.table across databases."
            )
            execution_records.append(record)
            continue

        if not is_safe:
            execution_records.append(record)
            continue

        try:
            result_df, target_errors = run_read_only_query_for_current_selection(
                sql_query=limited_sql,
                selected_database=selected_database,
                selected_table_targets=selected_table_targets,
                multi_database_sql_enabled=multi_database_sql_enabled,
            )
            record["result"] = result_df
            record["target_errors"] = target_errors
            record["status"] = f"Executed safely. Rows returned: {len(result_df)}"
        except Exception as e:
            record["error"] = str(e)
            record["status"] = f"Execution failed: {e}"

        execution_records.append(record)

    return execution_records


def build_agent_sql_result_prompt(
    agent_name,
    user_question,
    selected_table,
    selection_description,
    agent_answer,
    execution_records,
):
    result_sections = []

    for record in execution_records:
        result_df = record.get("result")
        if result_df is None:
            result_sections.append(
                f"""
Query {record["query_index"]} did not produce a result.
Status:
{record["status"]}

SQL:
{record["sql"]}
""".strip()
            )
            continue

        preview_df = result_df.head(60)
        result_sections.append(
            f"""
Query {record["query_index"]} result:
Rows returned: {len(result_df)}
Columns: {result_df.columns.tolist()}

SQL:
{record["sql"]}

Result preview:
{preview_df.to_string(index=False)}
""".strip()
        )

    results_text = "\n\n---\n\n".join(result_sections)

    return f"""
You are {agent_name}.

The user wanted the agent to answer the task using actual SQL results, not only
provide a SQL script.

User task:
{user_question}

Selected table:
{selected_table}

Selection:
{selection_description}

Your earlier agent report, for context:
{agent_answer[:6000]}

Executed read-only SQL results:
{results_text}

Now give the final answer to the user's task.

Rules:
- Answer directly first.
- Use the actual SQL result values.
- If the result has one row of KPIs, state those exact KPI values.
- If the result is grouped by user, database, or month, summarize the most important rows.
- Explain in beginner-friendly language.
- Mention if any query failed or if assumptions remain.
- Keep it practical and concise.
""".strip()


def build_global_agent_command_prompt(
    agent_name,
    agent_role,
    user_command,
    sql_server_memory_text,
    rag_context,
    remembered_tasks_text,
    allow_sql_execution,
):
    execution_text = (
        "Safe read-only SQL execution is enabled. If SQL is needed, write runnable SQL Server SELECT queries."
        if allow_sql_execution
        else "SQL execution is disabled. You may explain and suggest SQL, but do not rely on execution."
    )

    return f"""
You are {agent_name}.

Agent role:
{agent_role}

You are working inside a local/offline SQL Server analytics platform.
The user has not selected a database or table. You must use SQL Server context
memory to understand which databases, tables, and columns exist.

User command:
{user_command}

Execution mode:
{execution_text}

SQL Server context memory:
{sql_server_memory_text}

Relevant remembered agent tasks:
{remembered_tasks_text}

Retrieved RAG context:
{rag_context}

Rules:
- Answer the user's command directly.
- Do not search only exact words from the user's command. Think wider:
  use synonyms, business meaning, detected concepts, and key-like columns.
  Example: "paying users" may relate to Payment, PaymentMoney,
  PaymentContractKod, WaterUser, Contract, service user, payer, payment collection, or collection.
- Use a senior analytics workflow: frame the question, define metrics, identify
  data grain, segment results, validate assumptions, and explain business meaning.
- Use only SQL Server T-SQL.
- Keep all SQL read-only.
- Do not suggest INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, EXEC, USE, DBCC, or destructive commands.
- If you write executable SQL without a selected table, use fully qualified names such as [DatabaseName].[dbo].[Payment].
- For multi-database comparisons, use UNION ALL and create source labels as string literals:
  SELECT 'Structure_2024' AS SourceDatabase, ...
  Never write SourceDatabase or SourceTable by itself in global SQL because those
  are Python-added dataframe columns, not SQL Server table columns.
- Use only databases, tables, and columns that appear in SQL Server context memory.
- If the memory is not enough, say what metadata is missing.
- Do not claim a table does not exist unless the exact table name was checked in
  SQL Server context memory. If only the retrieved context is incomplete, say that.
- If the command asks for counts, totals, rankings, trends, or comparisons, provide SQL in a ```sql code block.
- If no SQL is needed, clearly explain the answer and recommend next steps.

Required answer format:

## Direct Answer
Answer the user's command in plain language.

## Analytics Frame
State business question, data grain, metrics, useful segments, and assumptions.

## Work Plan
Briefly explain how you approached the task.

## SQL Drafts
Provide read-only SQL Server SELECT queries if useful. Otherwise write: No SQL needed.

## Validation Notes
Mention assumptions, possible missing relationships, and how to verify.

## Executive Interpretation
Explain what a manager or interview audience should understand.

## Training Notes
Explain what analytics, SQL, BI, or data quality skill this teaches.
""".strip()


def format_remembered_tasks_for_prompt(search_text, limit=5, max_chars=5000):
    if MEMORY_IMPORT_ERROR:
        return "Agent memory is not available."

    tasks = load_agent_task_history(search_text=search_text, limit=limit)
    if not tasks:
        return "No remembered tasks matched this command."

    sections = []
    for task in tasks:
        sections.append(
            f"""
Saved at: {task.get("saved_at")}
Agent: {task.get("agent_name")}
Selection: {task.get("selection")}
Question: {safe_preview_text(task.get("question"), 600)}
Final answer: {safe_preview_text(task.get("final_answer"), 1000)}
SQL queries saved: {len(task.get("sql_execution_summaries") or [])}
""".strip()
        )

    text = "\n\n---\n\n".join(sections)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def execute_global_sql_queries(sql_queries):
    # Global commands do not have a selected table. To run successfully, generated
    # SQL should use fully qualified names such as [Structure_2025].[dbo].[Payment].
    execution_records = []

    for index, sql_query in enumerate(sql_queries[:AGENT_SQL_MAX_QUERIES], start=1):
        normalized_sql = normalize_sql_server_dialect(sql_query)
        limited_sql = add_top_limit_to_select(normalized_sql, AGENT_SQL_RESULT_LIMIT)
        is_safe, safety_message = validate_read_only_sql(limited_sql)

        record = {
            "query_index": index,
            "sql": limited_sql,
            "read_only_safe": bool(is_safe),
            "status": safety_message,
            "result": None,
            "target_errors": [],
            "error": None,
        }

        if has_bad_global_source_columns(limited_sql):
            record["read_only_safe"] = False
            record["status"] = (
                "Global SQL uses SourceDatabase or SourceTable as a bare column. "
                "Use a string literal instead, for example SELECT 'Structure_2025' AS SourceDatabase."
            )
            execution_records.append(record)
            continue

        if not is_safe:
            execution_records.append(record)
            continue

        try:
            result_df = run_read_only_query("master", limited_sql)
            record["result"] = result_df
            record["status"] = f"Executed safely from master connection. Rows returned: {len(result_df)}"
        except Exception as e:
            record["error"] = str(e)
            record["status"] = f"Execution failed: {e}"

        execution_records.append(record)

    return execution_records


def has_bad_global_source_columns(sql_query):
    # In loaded pandas results, SourceDatabase and SourceTable are Python-added
    # columns. In global SQL, they must be string literals, not bare identifiers.
    sql_without_string_literals = re.sub(r"'(?:''|[^'])*'", "''", str(sql_query))
    return bool(re.search(r"\bSourceDatabase\b|\bSourceTable\b", sql_without_string_literals))


def choose_needed_global_agents(user_command):
    command = str(user_command or "").lower()
    selected_agents = []

    def add_agent(agent_key):
        if agent_key not in selected_agents:
            selected_agents.append(agent_key)

    data_words = [
        "profile", "schema", "table", "column", "missing", "duplicate",
        "quality", "type", "metadata", "context", "all databases",
    ]
    analytics_lead_words = [
        "analyze", "analysis", "insight", "insights", "explain", "why",
        "strategy", "deep", "advanced", "top level", "business meaning",
        "executive", "manager", "story", "interpret", "decision",
    ]
    sql_words = [
        "sql", "query", "count", "sum", "total", "average", "top",
        "rank", "list", "find", "compare", "trend", "monthly",
        "group", "filter", "how many", "unique", "distinct",
    ]
    model_words = [
        "join", "relationship", "related", "foreign key", "primary key",
        "data model", "model", "star schema", "fact", "dimension",
        "connect tables", "link tables", "grain",
    ]
    anomaly_words = [
        "anomaly", "outlier", "unusual", "suspicious", "mismatch",
        "spike", "drop", "sudden", "abnormal", "duplicate behavior",
        "fraud", "exception", "high risk", "low payment",
    ]
    operations_words = [
        "wua", "payment", "payments", "collection", "water",
        "delivery", "service", "usage", "user", "customer",
        "anomaly", "contract", "quality", "segment",
    ]
    powerbi_words = [
        "power bi", "dashboard", "chart", "visual", "dax",
        "report", "page", "slicer", "kpi", "mis",
    ]
    rag_words = [
        "rag", "knowledge", "document", "policy", "manual",
        "explain from docs", "retrieved",
    ]

    if any(word in command for word in analytics_lead_words):
        add_agent("senior_analytics_lead")
    if any(word in command for word in data_words):
        add_agent("data_profiler")
    if any(word in command for word in model_words):
        add_agent("data_modeler")
    if any(word in command for word in sql_words) or question_needs_sql_result(command):
        add_agent("sql_analyst")
    if any(word in command for word in anomaly_words):
        add_agent("anomaly_detection")
    if any(word in command for word in operations_words):
        add_agent("risk_mis")
    if any(word in command for word in powerbi_words):
        add_agent("powerbi")
    if any(word in command for word in rag_words):
        add_agent("rag_knowledge")

    if not selected_agents:
        if question_needs_sql_result(command):
            add_agent("sql_analyst")
        else:
            add_agent("senior_analytics_lead")
            add_agent("data_profiler")

    add_agent("validator")
    return selected_agents


def build_global_coordinator_synthesis_prompt(
    user_command,
    sql_server_memory_text,
    rag_context,
    remembered_tasks_text,
    specialist_reports,
):
    report_sections = []

    for agent_key, report_text in specialist_reports.items():
        agent_name = AGENT_DEFINITIONS[agent_key]["name"]
        report_sections.append(
            f"### {agent_name}\n{str(report_text).strip()[:5000]}"
        )

    reports_text = "\n\n".join(report_sections)

    return f"""
You are the Auto Coordinator Agent for a local/offline SQL Server analytics platform.

The user gave one database-wide command. Python routed the command to the needed
specialist agents. Your job is to combine their work into one final answer and
keep the workflow practical, safe, and beginner-friendly.

User command:
{user_command}

SQL Server context memory:
{sql_server_memory_text}

Relevant remembered tasks:
{remembered_tasks_text}

Retrieved RAG context:
{rag_context}

Specialist reports:
{reports_text}

Rules:
- Answer the user's command directly.
- Do not rely only on exact names in the command. Use wider analysis:
  synonyms, table concepts, key-like columns, and business meaning.
- Use a senior analytics workflow: frame the business question, define metrics,
  identify data grain, segment the results, validate assumptions, and translate
  findings into executive meaning.
- If the specialists wrote SQL, choose the best read-only SQL drafts.
- Use SQL Server T-SQL only.
- Use fully qualified table names for executable global SQL, for example [Structure_2025].[dbo].[Payment].
- For multi-database comparisons, create source labels as string literals:
  SELECT 'Structure_2025' AS SourceDatabase, ...
  Never use SourceDatabase or SourceTable as if they are real SQL Server columns.
- Do not invent databases, tables, or columns.
- Do not claim a table does not exist unless the exact table name was checked in SQL Server context memory.
- Mention which specialist agents were used and why.
- Keep all SQL read-only.

Required answer format:

## Direct Answer
Give the best answer to the user's command.

## Analytics Frame
State business question, data grain, metrics, segments, and assumptions.

## Agents Used
List which agents were run and what each contributed.

## Best SQL Drafts
Include the best read-only SQL Server queries in ```sql code blocks if useful.

## Validation Notes
Explain assumptions and what should be checked.

## Executive Interpretation
Explain what a manager or interview audience should understand.

## Next Steps
Give practical next actions.
""".strip()


# -----------------------------
# LOCAL QWEN CONNECTION
# -----------------------------

def get_llm_client():
    return OpenAI(
        base_url=LMSTUDIO_URL,
        api_key="lm-studio",
        timeout=LMSTUDIO_TIMEOUT_SECONDS
    )


def get_lmstudio_models():
    client = get_llm_client()
    models = client.models.list()
    return [model.id for model in models.data]


def ask_qwen(prompt, max_tokens):
    client = get_llm_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior SQL Server, database analytics, and irrigation data analyst. "
                    "Answer directly and clearly for a beginner."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )

    choice = response.choices[0]
    message = choice.message
    finish_reason = choice.finish_reason
    answer = (message.content or "").strip()
    reasoning_text = getattr(message, "reasoning_content", "")

    # Some local Qwen reasoning models return text in reasoning_content first.
    # If the normal answer is blank, show a helpful fallback instead of nothing.
    if not answer:
        if reasoning_text:
            return (
                "LM Studio stopped Qwen before it produced a final answer. "
                "This usually means the prompt or the model's reasoning used the whole token budget. "
                "The app now sends a large automatic token budget, so increase context length / max prediction tokens in LM Studio if this repeats.\n\n"
                "Reasoning excerpt:\n"
                + reasoning_text.strip()[:1500]
            )

    if finish_reason == "length":
        answer += (
            "\n\nNote: LM Studio stopped this answer because it reached the token limit. "
            "If the answer looks incomplete, increase the model context / max prediction tokens in LM Studio."
        )

    return answer or "Qwen returned an empty response."


def repair_sql_query(original_sql, sql_error, selected_table, dtype_df, user_question):
    columns_text = dtype_df[["Column", "Data Type"]].to_string(index=False)

    repair_prompt = f"""
Fix this SQL Server SELECT query.

The previous query failed with this SQL Server error:
{sql_error}

Selected table:
{selected_table}

Available columns:
{columns_text}

Original user question:
{user_question}

Failed SQL:
{original_sql}

Rules:
- Return only one corrected SQL Server SELECT query.
- Put it inside a ```sql code block.
- The code block must contain only runnable SQL.
- Use SQL Server T-SQL only. Do not use DATE_TRUNC, LIMIT, ILIKE, or PostgreSQL casts.
- For monthly trends, use DATEFROMPARTS(YEAR(date_column), MONTH(date_column), 1).
- Use the exact selected table name: {selected_table}
- Use only columns from the available columns list.
- Do not use English words from the user question as column names unless they appear in the available columns list.
- Do not include explanation, labels, headings, comments, or markdown outside the SQL code block.
"""

    repair_answer = ask_qwen(repair_prompt, max_tokens=3000)
    return extract_sql_query(repair_answer), repair_answer


def question_needs_sql_result(user_question):
    question = user_question.lower()
    sql_intent_words = [
        "sql",
        "query",
        "show",
        "get",
        "list",
        "find",
        "filter",
        "sort",
        "rank",
        "top",
        "bottom",
        "count",
        "sum",
        "total",
        "average",
        "avg",
        "min",
        "max",
        "trend",
        "chart",
        "line chart",
        "bar chart",
        "revenue",
        "payment",
        "payments",
        "over time",
        "by month",
        "monthly",
        "yearly",
        "group by",
    ]
    return any(word in question for word in sql_intent_words)


def ask_qwen_for_sql_only(user_question, selected_table, dtype_df, rag_context):
    columns_text = dtype_df[["Column", "Data Type"]].to_string(index=False)

    sql_only_prompt = f"""
Write one runnable SQL Server SELECT query for the user's request.

Selected table:
{selected_table}

Available columns:
{columns_text}

Retrieved RAG context, if useful:
{rag_context}

User request:
{user_question}

Rules:
- Return only one SQL Server SELECT query.
- Put it inside a ```sql code block.
- The code block must contain only runnable SQL.
- Use the exact selected table name: {selected_table}
- Use only columns from the available columns list.
- Do not invent columns.
- Do not use DATE_TRUNC, LIMIT, ILIKE, PostgreSQL casts, or non-SQL Server syntax.
- For monthly trends, use DATEFROMPARTS(YEAR(date_column), MONTH(date_column), 1).
- Add TOP 100 or a smaller TOP value unless the user asks for a different number.
- Do not include explanation, labels, headings, comments, or markdown outside the SQL code block.
"""

    sql_only_answer = ask_qwen(sql_only_prompt, max_tokens=3000)
    return extract_sql_query(sql_only_answer), sql_only_answer


def check_lmstudio_ready():
    models = get_lmstudio_models()
    if MODEL_NAME not in models:
        raise ValueError(
            f"LM Studio is running, but model '{MODEL_NAME}' was not found. "
            f"Available models: {models}"
        )
    return models


def choose_ai_context_size(df):
    # Automatic sizing: use more context for small tables and shrink gently for wide tables.
    row_count = len(df)
    column_count = len(df.columns)

    if row_count == 0:
        sample_rows = 0
    elif column_count <= 20:
        sample_rows = min(row_count, 25)
    elif column_count <= 60:
        sample_rows = min(row_count, 15)
    elif column_count <= 120:
        sample_rows = min(row_count, 8)
    else:
        sample_rows = min(row_count, 5)

    if column_count <= 30:
        max_columns = column_count
        max_cell_length = 300
    elif column_count <= 100:
        max_columns = column_count
        max_cell_length = 180
    else:
        max_columns = min(column_count, 150)
        max_cell_length = 120

    return max_columns, sample_rows, max_cell_length


def safe_text_value(value, max_length):
    # SQL Server tables sometimes contain binary values or old Windows-encoded text.
    # Convert them safely so AI context building never crashes on Unicode decoding.
    if value is None:
        return "<missing>"

    try:
        missing_value = pd.isna(value)
        if isinstance(missing_value, bool) and missing_value:
            return "<missing>"
    except Exception:
        pass

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        raw_value = bytes(value)
        decoded_value = None

        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                decoded_value = raw_value.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_value is None:
            decoded_value = repr(raw_value)

        text = decoded_value
    else:
        try:
            text = str(value)
        except Exception:
            text = repr(value)

    text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    return text[:max_length]


def build_ai_table_context(df, dtype_df):
    max_columns, sample_rows, max_cell_length = choose_ai_context_size(df)
    ai_columns = df.columns.tolist()[:max_columns]
    sample_df = df[ai_columns].head(sample_rows).copy()

    for column in sample_df.columns:
        sample_df[column] = sample_df[column].map(
            lambda value: safe_text_value(value, max_cell_length)
        )

    dtype_for_ai = dtype_df[dtype_df["Column"].isin(ai_columns)]
    missing_for_ai = dtype_df[dtype_df["Missing Values"] > 0][
        ["Column", "Missing Values", "Missing %"]
    ].head(max_columns)

    all_columns = df.columns.tolist()
    omitted_columns = max(len(all_columns) - len(ai_columns), 0)
    sample_rows_text = (
        sample_df.to_string(index=False)
        if len(sample_df) > 0
        else "No sample rows available because the loaded table returned 0 rows."
    )

    context = {
        "all_columns_text": all_columns[:300],
        "included_columns": ai_columns,
        "omitted_columns": omitted_columns,
        "dtypes_text": dtype_for_ai.to_string(index=False),
        "missing_text": (
            missing_for_ai.to_string(index=False)
            if len(missing_for_ai) > 0
            else "No missing values in loaded rows."
        ),
        "sample_rows": sample_rows_text,
        "sample_row_count": len(sample_df),
        "max_cell_length": max_cell_length,
        "max_tokens": AUTO_QWEN_MAX_TOKENS,
    }

    # If the table is very wide, automatically shrink the sample until the prompt stays usable.
    # This avoids manual sliders while still giving Qwen as much context as practical.
    while len(str(context)) > AUTO_PROMPT_CHAR_LIMIT and len(ai_columns) > 10:
        ai_columns = ai_columns[: max(10, int(len(ai_columns) * 0.75))]
        sample_rows = max(1, int(sample_rows * 0.75)) if sample_rows > 0 else 0
        max_cell_length = max(80, int(max_cell_length * 0.75))

        sample_df = df[ai_columns].head(sample_rows).copy()
        for column in sample_df.columns:
            sample_df[column] = sample_df[column].map(
                lambda value: safe_text_value(value, max_cell_length)
            )

        dtype_for_ai = dtype_df[dtype_df["Column"].isin(ai_columns)]
        missing_for_ai = dtype_df[dtype_df["Missing Values"] > 0][
            ["Column", "Missing Values", "Missing %"]
        ].head(len(ai_columns))

        context.update({
            "included_columns": ai_columns,
            "omitted_columns": max(len(all_columns) - len(ai_columns), 0),
            "dtypes_text": dtype_for_ai.to_string(index=False),
            "missing_text": (
                missing_for_ai.to_string(index=False)
                if len(missing_for_ai) > 0
                else "No missing values in loaded rows."
            ),
            "sample_rows": (
                sample_df.to_string(index=False)
                if len(sample_df) > 0
                else "No sample rows available because the loaded table returned 0 rows."
            ),
            "sample_row_count": len(sample_df),
            "max_cell_length": max_cell_length,
        })

    return context


def build_agent_workspace_evidence(
    df,
    dtype_df,
    selected_database,
    selected_table,
    selection_description,
    selected_table_targets,
):
    ai_context = build_ai_table_context(df, dtype_df)

    selected_targets_text = (
        pd.DataFrame(selected_table_targets).to_string(index=False)
        if selected_table_targets
        else "One selected table target."
    )

    risk_outputs = st.session_state.get("risk_training_outputs")
    risk_summary_text = "No optional operational BI training output has been generated yet."
    if risk_outputs:
        risk_summary_text = risk_outputs["risk_training_summary_df"].head(80).to_string(index=False)

    powerbi_files = st.session_state.get("risk_training_file_paths", {})
    powerbi_pack_files = st.session_state.get("powerbi_automation_pack_paths", {})
    all_powerbi_files = {**powerbi_files, **powerbi_pack_files}
    powerbi_files_text = (
        "\n".join(f"{label}: {path}" for label, path in all_powerbi_files.items())
        if all_powerbi_files
        else "No Power BI output files have been generated yet."
    )

    selected_target_count = len(selected_table_targets)
    selected_full_table_names = sorted({
        target.get("FULL_TABLE_NAME", "")
        for target in selected_table_targets
        if target.get("FULL_TABLE_NAME")
    })
    agent_sql_execution_enabled = selected_target_count == 1 or len(selected_full_table_names) == 1
    agent_multi_database_sql_enabled = selected_target_count > 1 and len(selected_full_table_names) == 1

    if selected_target_count == 1:
        agent_sql_rules_text = f"""
- The Agent Workspace can execute safe read-only SQL for this selected table.
- Use the exact selected table name: {selected_table}
- Use SQL Server T-SQL only.
- Use only columns listed in the evidence.
- For row lists, use TOP 100 or less unless the user asks for another number.
""".strip()
    elif agent_multi_database_sql_enabled:
        agent_sql_rules_text = f"""
- The Agent Workspace can execute the same safe read-only SQL inside each selected database.
- Use the exact table name: {selected_table}
- Do not prefix the table with a database name.
- Do not use SourceDatabase or SourceTable inside the SQL; Python adds those columns after each database returns results.
- Use SQL Server T-SQL only.
- Use only columns listed in the evidence.
- For row lists, use TOP 100 or less unless the user asks for another number.
""".strip()
    else:
        agent_sql_rules_text = """
- The Agent Workspace cannot automatically execute SQL for this selection because multiple different table targets are loaded.
- Agents may provide SQL templates, but final execution needs one exact table or the same schema.table across selected databases.
- If analyzing the loaded dataframe conceptually, use SourceDatabase and SourceTable only in the explanation, not as SQL Server columns.
""".strip()

    return {
        "selection_description": selection_description,
        "selected_database": selected_database,
        "selected_table": selected_table,
        "row_count": len(df),
        "column_count": len(df.columns),
        "selected_targets_text": selected_targets_text,
        "columns_text": ai_context["all_columns_text"],
        "dtypes_text": ai_context["dtypes_text"],
        "missing_text": ai_context["missing_text"],
        "sample_rows": ai_context["sample_rows"],
        "risk_summary_text": risk_summary_text,
        "powerbi_files_text": powerbi_files_text,
        "sql_execution_enabled": agent_sql_execution_enabled,
        "multi_database_sql_enabled": agent_multi_database_sql_enabled,
        "sql_rules_text": agent_sql_rules_text,
        "sql_server_memory_text": "SQL Server context memory was not used for this agent run.",
        "ai_context": ai_context,
    }


def find_column(df, candidate_names):
    # SQL systems often use slightly different names for the same business idea.
    # This helper finds the first matching column without caring about case.
    column_lookup = {str(column).lower(): column for column in df.columns}

    for candidate_name in candidate_names:
        matched_column = column_lookup.get(candidate_name.lower())
        if matched_column is not None:
            return matched_column

    return None


def missing_or_blank_count(series):
    # For BI/data quality work, blanks such as "" should count like missing values.
    text_values = series.astype("string").str.strip()
    return int(series.isna().sum() + text_values.eq("").sum())


def missing_or_blank_rate(series):
    if len(series) == 0:
        return 0.0

    return round((missing_or_blank_count(series) / len(series)) * 100, 2)


def safe_numeric_sum(series):
    numeric_values = pd.to_numeric(series, errors="coerce")
    return float(numeric_values.fillna(0).sum())


def clean_dataframe_for_powerbi(df):
    # Power BI likes clean UTF-8 CSV files. Convert old binary/text values safely
    # while leaving numbers and dates available for BI visuals.
    clean_df = df.copy()

    for column in clean_df.columns:
        if clean_df[column].dtype == "object" or str(clean_df[column].dtype).startswith("string"):
            clean_df[column] = clean_df[column].map(
                lambda value: safe_text_value(value, 1000)
            )

    return clean_df


def build_risk_mis_outputs(df):
    total_records = len(df)
    duplicate_rows = int(df.duplicated().sum()) if total_records else 0
    duplicate_row_rate = round((duplicate_rows / total_records) * 100, 2) if total_records else 0.0

    wua_col = find_column(df, ["WuaID", "WUAID", "WuaId"])
    phone_col = find_column(df, ["WaterUserPhone", "Phone", "PhoneNumber", "MobilePhone"])
    address_col = find_column(df, ["WaterUserAddress", "Address", "UserAddress"])
    user_code_col = find_column(df, ["WaterUserKod", "WaterUserCode", "UserCode", "CustomerCode"])
    enter_date_col = find_column(df, ["EnterDateTime", "CreatedDate", "RegistrationDate", "CreateDate"])
    payment_col = find_column(df, ["PaymentMoney", "PaymentAmount", "PaidAmount", "AmountPaid", "CollectedAmount"])
    debt_col = find_column(df, ["Debt", "DebtMoney", "DebtAmount", "UnpaidAmount", "OverdueAmount", "Balance"])
    water_col = find_column(df, ["WaterDelivered", "DeliveredWater", "WaterAmount", "WaterVolume", "GivenWater", "Water"])

    missing_total = int(df.isna().sum().sum())
    total_cells = int(total_records * len(df.columns)) if len(df.columns) else 0
    overall_missing_rate = round((missing_total / total_cells) * 100, 2) if total_cells else 0.0

    data_quality_df = pd.DataFrame({
        "Column": df.columns,
        "DataType": [str(dtype) for dtype in df.dtypes],
        "MissingValues": df.isna().sum().values,
        "MissingRatePct": (
            ((df.isna().sum() / total_records) * 100).round(2).values
            if total_records
            else [0] * len(df.columns)
        ),
        "UniqueValues": [df[column].nunique(dropna=True) for column in df.columns],
    })

    data_quality_df["RiskFlag"] = data_quality_df["MissingRatePct"].apply(
        lambda value: "High missingness" if value >= 30 else ("Medium missingness" if value >= 10 else "OK")
    )

    unique_wua_count = int(df[wua_col].nunique(dropna=True)) if wua_col else None
    missing_phone_rate = missing_or_blank_rate(df[phone_col]) if phone_col else None
    missing_address_rate = missing_or_blank_rate(df[address_col]) if address_col else None

    duplicate_user_code_rows = None
    duplicate_user_code_count = None
    duplicate_identity_rate = 0.0
    if user_code_col:
        user_codes = df[user_code_col].dropna().astype("string").str.strip()
        user_codes = user_codes[user_codes != ""]
        duplicate_user_code_rows = int(user_codes.duplicated(keep=False).sum())
        duplicate_user_code_count = int((user_codes.value_counts() > 1).sum())
        duplicate_identity_rate = round((duplicate_user_code_rows / total_records) * 100, 2) if total_records else 0.0

    records_by_wua_df = pd.DataFrame(columns=["WuaID", "RecordCount", "PortfolioSharePct"])
    top_wua_share = 0.0
    if wua_col and total_records:
        records_by_wua_df = (
            df[wua_col]
            .fillna("<missing>")
            .value_counts(dropna=False)
            .reset_index()
        )
        records_by_wua_df.columns = ["WuaID", "RecordCount"]
        records_by_wua_df["PortfolioSharePct"] = (
            (records_by_wua_df["RecordCount"] / total_records) * 100
        ).round(2)
        top_wua_share = float(records_by_wua_df["PortfolioSharePct"].max()) if not records_by_wua_df.empty else 0.0

    registration_trend_df = pd.DataFrame(columns=["RegistrationMonth", "RecordCount"])
    if enter_date_col:
        registration_dates = pd.to_datetime(df[enter_date_col], errors="coerce")
        registration_trend_df = (
            registration_dates
            .dt.to_period("M")
            .astype("string")
            .value_counts()
            .sort_index()
            .reset_index()
        )
        registration_trend_df.columns = ["RegistrationMonth", "RecordCount"]

    missing_contact_risk = max(
        value for value in [missing_phone_rate or 0.0, missing_address_rate or 0.0]
    )

    payment_collection_risk = None
    payment_collection_note = "Payment/debt columns were not detected."
    if payment_col and debt_col:
        total_paid = safe_numeric_sum(df[payment_col])
        total_debt = safe_numeric_sum(df[debt_col])
        denominator = abs(total_paid) + abs(total_debt)
        payment_collection_risk = round((abs(total_debt) / denominator) * 100, 2) if denominator else 0.0
        payment_collection_note = (
            f"Debt-like column `{debt_col}` compared with payment-like column `{payment_col}`."
        )
    elif payment_col:
        payment_values = pd.to_numeric(df[payment_col], errors="coerce")
        payment_collection_risk = round(
            ((payment_values.isna() | (payment_values <= 0)).sum() / total_records) * 100,
            2,
        ) if total_records else 0.0
        payment_collection_note = (
            f"No debt column found. Risk proxy uses missing/zero `{payment_col}` rows."
        )

    water_payment_mismatch_rate = None
    water_payment_note = "Water-delivery and payment columns were not both detected."
    if water_col and payment_col:
        water_values = pd.to_numeric(df[water_col], errors="coerce")
        payment_values = pd.to_numeric(df[payment_col], errors="coerce")
        mismatch_mask = (water_values > 0) & (payment_values.isna() | (payment_values <= 0))
        water_payment_mismatch_rate = round((mismatch_mask.sum() / total_records) * 100, 2) if total_records else 0.0
        water_payment_note = (
            f"Rows with `{water_col}` > 0 and missing/zero `{payment_col}`."
        )

    # Training score only: this is not a regulatory model.
    data_quality_risk_score = min(
        100.0,
        round(
            (overall_missing_rate * 0.40)
            + (duplicate_row_rate * 0.20)
            + (missing_contact_risk * 0.20)
            + (duplicate_identity_rate * 0.10)
            + (top_wua_share * 0.10),
            2,
        ),
    )

    kpi_rows = [
        ("Total records", total_records, "Rows loaded from SQL Server"),
        ("Column count", len(df.columns), "Number of fields in selected table"),
        ("Missing values total", missing_total, "Blank/null cells across loaded rows"),
        ("Duplicate rows", duplicate_rows, "Exact duplicate dataframe rows"),
        ("Duplicate row rate %", duplicate_row_rate, "Duplicate rows divided by total rows"),
        ("Unique WUA count", unique_wua_count if unique_wua_count is not None else "N/A", "Available if WuaID exists"),
        ("Missing phone rate %", missing_phone_rate if missing_phone_rate is not None else "N/A", "Available if WaterUserPhone exists"),
        ("Missing address rate %", missing_address_rate if missing_address_rate is not None else "N/A", "Available if WaterUserAddress exists"),
        ("Duplicate user code rows", duplicate_user_code_rows if duplicate_user_code_rows is not None else "N/A", "Available if WaterUserKod exists"),
        ("Duplicate user code count", duplicate_user_code_count if duplicate_user_code_count is not None else "N/A", "Distinct duplicated user codes"),
    ]
    kpi_summary_df = pd.DataFrame(kpi_rows, columns=["Metric", "Value", "Notes"])

    risk_rows = [
        ("Readiness Score", "Data quality readiness score", data_quality_risk_score, "0-100 training score from missingness, duplicates, contact gaps, identity duplication, concentration"),
        ("Contact Completeness", "Missing contact rate %", missing_contact_risk, "Maximum of missing phone and missing address rates"),
        ("Identity Quality", "Duplicate identity rate %", duplicate_identity_rate, "Duplicate WaterUserKod proxy where available"),
        ("Segment Concentration", "Top WUA segment share %", round(top_wua_share, 2), "Largest WUA share of loaded records"),
        ("Payment Collection", "Estimated payment collection gap %", payment_collection_risk if payment_collection_risk is not None else "N/A", payment_collection_note),
        ("Usage/Payment Mismatch", "Water delivery vs payment mismatch %", water_payment_mismatch_rate if water_payment_mismatch_rate is not None else "N/A", water_payment_note),
    ]

    for _, row in records_by_wua_df.head(50).iterrows():
        risk_rows.append(
            (
                "Segment Concentration",
                f"WUA {row['WuaID']}",
                row["RecordCount"],
                f"Segment share: {row['PortfolioSharePct']}%",
            )
        )

    for _, row in registration_trend_df.head(120).iterrows():
        risk_rows.append(
            (
                "Registration Trend",
                row["RegistrationMonth"],
                row["RecordCount"],
                "Records registered in this month",
            )
        )

    risk_training_summary_df = pd.DataFrame(
        risk_rows,
        columns=["Category", "Metric", "Value", "Interpretation"],
    )

    return {
        "clean_df": clean_dataframe_for_powerbi(df),
        "kpi_summary_df": kpi_summary_df,
        "data_quality_df": data_quality_df,
        "risk_training_summary_df": risk_training_summary_df,
        "records_by_wua_df": records_by_wua_df,
        "registration_trend_df": registration_trend_df,
        "detected_columns": {
            "WuaID": wua_col,
            "WaterUserPhone": phone_col,
            "WaterUserAddress": address_col,
            "WaterUserKod": user_code_col,
            "EnterDateTime": enter_date_col,
            "Payment": payment_col,
            "Debt": debt_col,
            "WaterDelivery": water_col,
        },
        "risk_score": data_quality_risk_score,
    }


def save_powerbi_outputs(outputs):
    os.makedirs(POWERBI_OUTPUT_DIR, exist_ok=True)

    file_paths = {
        "clean_table_export": os.path.join(POWERBI_OUTPUT_DIR, "clean_table_export.csv"),
        "kpi_summary": os.path.join(POWERBI_OUTPUT_DIR, "kpi_summary.csv"),
        "data_quality_summary": os.path.join(POWERBI_OUTPUT_DIR, "data_quality_summary.csv"),
        "risk_training_summary": os.path.join(POWERBI_OUTPUT_DIR, "risk_training_summary.csv"),
    }

    outputs["clean_df"].to_csv(file_paths["clean_table_export"], index=False, encoding="utf-8-sig")
    outputs["kpi_summary_df"].to_csv(file_paths["kpi_summary"], index=False, encoding="utf-8-sig")
    outputs["data_quality_df"].to_csv(file_paths["data_quality_summary"], index=False, encoding="utf-8-sig")
    outputs["risk_training_summary_df"].to_csv(file_paths["risk_training_summary"], index=False, encoding="utf-8-sig")

    return file_paths


def infer_powerbi_semantic_role(column_name, detected_columns):
    # These roles help Power BI users understand what each field is likely for.
    # They are training labels, not hard business rules.
    if column_name == "SourceDatabase":
        return "Source database / comparison group"
    if column_name == "SourceTable":
        return "Source table lineage"
    if column_name == detected_columns.get("WuaID"):
        return "Operational group / WUA / segment"
    if column_name == detected_columns.get("WaterUserPhone"):
        return "Service user contact completeness"
    if column_name == detected_columns.get("WaterUserAddress"):
        return "Service user address completeness"
    if column_name == detected_columns.get("WaterUserKod"):
        return "Service user / identity key"
    if column_name == detected_columns.get("EnterDateTime"):
        return "Registration / onboarding date"
    if column_name == detected_columns.get("Payment"):
        return "Payment collection amount"
    if column_name == detected_columns.get("Debt"):
        return "Unpaid service amount"
    if column_name == detected_columns.get("WaterDelivery"):
        return "Service usage"

    lowered_name = str(column_name).lower()
    if "date" in lowered_name or "time" in lowered_name:
        return "Date / time field"
    if "id" in lowered_name or "kod" in lowered_name or "code" in lowered_name:
        return "Identifier"
    if "money" in lowered_name or "amount" in lowered_name or "sum" in lowered_name:
        return "Amount / measure"

    return "Descriptive attribute"


def build_powerbi_data_dictionary(df, outputs):
    data_quality_df = outputs["data_quality_df"]
    detected_columns = outputs["detected_columns"]

    dictionary_df = data_quality_df.copy()
    dictionary_df["PowerBIRole"] = dictionary_df["Column"].apply(
        lambda column: infer_powerbi_semantic_role(column, detected_columns)
    )
    dictionary_df["RecommendedVisualUse"] = dictionary_df["PowerBIRole"].apply(
        lambda role: (
            "Axis, slicer, or legend"
            if role in [
                "Operational group / WUA / segment",
                "Date / time field",
                "Descriptive attribute",
                "Source database / comparison group",
                "Source table lineage",
            ]
            else "KPI, table, or data quality rule"
        )
    )
    dictionary_df["TrainingNote"] = dictionary_df["Column"].apply(
        lambda column: (
            "Review this field carefully because it has high missingness."
            if float(
                data_quality_df.loc[data_quality_df["Column"] == column, "MissingRatePct"].iloc[0]
            ) >= 30
            else "Available for Power BI modeling."
        )
    )

    return dictionary_df


def build_powerbi_risk_kpi_definitions(outputs):
    detected_columns = outputs["detected_columns"]
    payment_col = detected_columns.get("Payment") or "PaymentMoney"
    water_col = detected_columns.get("WaterDelivery") or "WaterDelivered"
    wua_col = detected_columns.get("WuaID") or "WuaID"

    rows = [
        {
            "KPIName": "Total Records",
            "BusinessMeaning": "Dataset size / number of records in the selected training table.",
            "DAXMeasure": "Total Records = COUNTROWS(clean_table_export)",
            "RecommendedVisual": "Card",
        },
        {
            "KPIName": "Data Quality Readiness Score",
            "BusinessMeaning": "Training score combining missingness, duplicates, contact gaps, identity duplication, and concentration.",
            "DAXMeasure": "Data Quality Readiness Score = MAXX(FILTER(risk_training_summary, risk_training_summary[Metric] = \"Data quality readiness score\"), VALUE(risk_training_summary[Value]))",
            "RecommendedVisual": "Card / gauge",
        },
        {
            "KPIName": "Missing Values",
            "BusinessMeaning": "Total missing values across the exported table.",
            "DAXMeasure": "Missing Values = SUM(data_quality_summary[MissingValues])",
            "RecommendedVisual": "Card / bar chart",
        },
        {
            "KPIName": "Average Missing Rate",
            "BusinessMeaning": "Average missing-value rate across columns.",
            "DAXMeasure": "Average Missing Rate = AVERAGE(data_quality_summary[MissingRatePct])",
            "RecommendedVisual": "Card",
        },
        {
            "KPIName": "High Missing Columns",
            "BusinessMeaning": "Count of fields with missing rate of 30% or higher.",
            "DAXMeasure": "High Missing Columns = COUNTROWS(FILTER(data_quality_summary, data_quality_summary[MissingRatePct] >= 30))",
            "RecommendedVisual": "Card / table",
        },
        {
            "KPIName": "Operational Groups",
            "BusinessMeaning": "Number of WUA groups or operational segments.",
            "DAXMeasure": f"Operational Groups = DISTINCTCOUNT(clean_table_export[{wua_col}])",
            "RecommendedVisual": "Card / slicer",
        },
        {
            "KPIName": "Total Payments",
            "BusinessMeaning": "Payment collection amount.",
            "DAXMeasure": f"Total Payments = SUM(clean_table_export[{payment_col}])",
            "RecommendedVisual": "Card / line chart",
        },
        {
            "KPIName": "Total Water Delivered",
            "BusinessMeaning": "Service usage amount.",
            "DAXMeasure": f"Total Water Delivered = SUM(clean_table_export[{water_col}])",
            "RecommendedVisual": "Card / scatter chart",
        },
    ]

    return pd.DataFrame(rows)


def build_powerbi_measures_dax(outputs):
    detected_columns = outputs["detected_columns"]
    wua_col = detected_columns.get("WuaID") or "WuaID"
    payment_col = detected_columns.get("Payment") or "PaymentMoney"
    water_col = detected_columns.get("WaterDelivery") or "WaterDelivered"

    # This is a helper file for Power BI Desktop. Paste measures into a report,
    # then remove or edit any measure whose source column does not exist.
    return f"""
-- Power BI DAX measures generated locally by Streamlit.
-- Paste these into Power BI Desktop after importing the CSV files.
-- Edit column names if your table uses different names.

Total Records =
COUNTROWS(clean_table_export)

Missing Values =
SUM(data_quality_summary[MissingValues])

Average Missing Rate =
AVERAGE(data_quality_summary[MissingRatePct])

High Missing Columns =
COUNTROWS(
    FILTER(data_quality_summary, data_quality_summary[MissingRatePct] >= 30)
)

Data Quality Readiness Score =
MAXX(
    FILTER(risk_training_summary, risk_training_summary[Metric] = "Data quality readiness score"),
    VALUE(risk_training_summary[Value])
)

Missing Contact Rate =
MAXX(
    FILTER(risk_training_summary, risk_training_summary[Metric] = "Missing contact rate %"),
    VALUE(risk_training_summary[Value])
)

Duplicate Identity Rate =
MAXX(
    FILTER(risk_training_summary, risk_training_summary[Metric] = "Duplicate identity rate %"),
    VALUE(risk_training_summary[Value])
)

Top WUA Segment Share =
MAXX(
    FILTER(risk_training_summary, risk_training_summary[Metric] = "Top WUA segment share %"),
    VALUE(risk_training_summary[Value])
)

Operational Groups =
DISTINCTCOUNT(clean_table_export[{wua_col}])

Total Payments =
SUM(clean_table_export[{payment_col}])

Total Water Delivered =
SUM(clean_table_export[{water_col}])

Payment Per Water Unit =
DIVIDE([Total Payments], [Total Water Delivered])
""".strip()


def build_power_query_import_scripts():
    output_folder = os.path.abspath(POWERBI_OUTPUT_DIR).replace("\\", "\\\\")

    # Power Query M can import all generated CSVs from the local output folder.
    return f"""
// Power Query M import scripts generated locally.
// In Power BI Desktop, use Get Data -> Blank Query -> Advanced Editor.
// Update PowerBIOutputPath if you move the output folder.

let
    PowerBIOutputPath = "{output_folder}\\\\",

    clean_table_export =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "clean_table_export.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        ),

    kpi_summary =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "kpi_summary.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        ),

    data_quality_summary =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "data_quality_summary.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        ),

    risk_training_summary =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "risk_training_summary.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        ),

    data_dictionary =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "data_dictionary.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        ),

    risk_kpi_definitions =
        Table.PromoteHeaders(
            Csv.Document(
                File.Contents(PowerBIOutputPath & "risk_kpi_definitions.csv"),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            [PromoteAllScalars=true]
        )
in
    clean_table_export
""".strip()


def build_powerbi_sql_views(database_name, full_table_name, outputs, table_targets=None):
    detected_columns = outputs["detected_columns"]
    payment_col = detected_columns.get("Payment") or "PaymentMoney"
    enter_date_col = detected_columns.get("EnterDateTime") or "EnterDateTime"

    if table_targets and len(table_targets) > 1:
        unique_full_table_names = sorted(
            {target["FULL_TABLE_NAME"] for target in table_targets}
        )

        if len(unique_full_table_names) > 1:
            select_templates = []

            for target in table_targets:
                target_schema, target_table = target["FULL_TABLE_NAME"].split(".", 1)
                select_templates.append(
                    f"""-- {target["DISPLAY_TABLE_NAME"]}
SELECT TOP 100
    '{target["DATABASE_NAME"]}' AS SourceDatabase,
    '{target["FULL_TABLE_NAME"]}' AS SourceTable,
    *
FROM [{target["DATABASE_NAME"]}].[{target_schema}].[{target_table}];"""
                )

            return f"""
-- SQL Server helper script for multiple exact table targets.
-- This file is NOT executed by Streamlit.
-- Review it manually before running in SQL Server Management Studio.
--
-- The selected targets are different schema.table names.
-- A single UNION ALL query may fail if their columns are not identical.
-- Streamlit exports them safely through pandas with SourceDatabase and SourceTable columns.
-- Use the individual templates below for manual SQL exploration.

{chr(10).join(select_templates)}
""".strip()

        schema_name, table_name = unique_full_table_names[0].split(".", 1)
        union_parts = []
        for target in table_targets:
            union_parts.append(
                f"""SELECT
    '{target["DATABASE_NAME"]}' AS SourceDatabase,
    '{target["FULL_TABLE_NAME"]}' AS SourceTable,
    *
FROM [{target["DATABASE_NAME"]}].[{schema_name}].[{table_name}]"""
            )

        union_query = "\nUNION ALL\n".join(union_parts)

        return f"""
-- SQL Server helper script for a multi-database Power BI comparison.
-- This file is NOT executed by Streamlit.
-- Review it manually before running in SQL Server Management Studio.
-- The selected table is `{full_table_name}` across multiple databases.

-- Option A: use this UNION ALL query in Power BI native SQL.
{union_query};

-- Option B: create a view manually in a reporting database after reviewing permissions.
-- CREATE OR ALTER VIEW [dbo].[vw_PowerBI_{table_name}_MultiDatabase] AS
-- {union_query.replace(chr(10), chr(10) + "-- ")};
""".strip()

    if "." not in full_table_name:
        return f"""
-- SQL Server helper script was not generated as a single view.
-- Selected source: {database_name}
-- Selected table label: {full_table_name}
-- Reason: no single schema.table target was available.
-- Use the exported CSV files for Power BI, or regenerate from one exact table.
""".strip()

    schema_name, table_name = full_table_name.split(".", 1)

    return f"""
-- SQL Server helper script for Power BI.
-- This file is NOT executed by Streamlit.
-- Review it manually before running in SQL Server Management Studio.
-- Database: {database_name}

USE [{database_name}];
GO

CREATE OR ALTER VIEW [dbo].[vw_PowerBI_{table_name}_Clean] AS
SELECT *
FROM [{schema_name}].[{table_name}];
GO

-- Monthly payment-style trend view.
-- Edit column names if your table uses different payment/date fields.
CREATE OR ALTER VIEW [dbo].[vw_PowerBI_{table_name}_MonthlyTrend] AS
SELECT
    DATEFROMPARTS(YEAR([{enter_date_col}]), MONTH([{enter_date_col}]), 1) AS TrendMonth,
    COUNT(*) AS RecordCount,
    SUM(TRY_CAST([{payment_col}] AS float)) AS TotalPaymentAmount
FROM [{schema_name}].[{table_name}]
WHERE [{enter_date_col}] IS NOT NULL
GROUP BY DATEFROMPARTS(YEAR([{enter_date_col}]), MONTH([{enter_date_col}]), 1);
GO
""".strip()


def build_dashboard_blueprint(database_name, full_table_name, outputs, table_targets=None):
    detected_columns = outputs["detected_columns"]

    return f"""
# Power BI Dashboard Blueprint

Generated locally for:

- Database: `{database_name}`
- Table: `{full_table_name}`
- Table targets: `{len(table_targets) if table_targets else 1}`
- Data quality readiness score: `{outputs["risk_score"]}`

## Business Framing

Use the irrigation/WUA dataset as an operational BI training dataset:

- Water users = service users
- WUA = organization unit or segment
- Water delivered = service usage
- Payment collected = payment collection
- Unpaid amount / debt = unpaid service amount
- Missing contact/address/identity fields = data quality issue

## Detected Columns

```text
{json.dumps(detected_columns, indent=2)}
```

## Page 1: Executive Overview

Purpose: show dataset size and headline data readiness.

Recommended visuals:

- Total Records card
- Data Quality Readiness Score card
- Missing Values card
- Average Missing Rate card
- High Missing Columns card

## Page 2: Data Quality Readiness

Purpose: show which fields need cleanup for BI reporting.

Recommended visuals:

- Bar chart: `data_quality_summary[Column]` by `MissingRatePct`
- Table: columns with RiskFlag = High missingness
- KPI card: High Missing Columns

## Page 3: WUA / Segment Concentration

Purpose: show whether records are concentrated in one WUA / segment.

Recommended visuals:

- Bar chart: WUA by record count
- Treemap: WUA segment share
- Card: Top WUA Segment Share

If multiple databases were selected, add `SourceDatabase` as a slicer or legend.
If multiple exact tables were selected, add `SourceTable` as a slicer or legend.

## Page 4: Payment vs Water Delivery

Purpose: compare service usage and collection behavior.

Recommended visuals:

- Line chart: monthly payment trend
- Scatter chart: water delivered vs payment amount
- Table: high usage with zero/missing payment

## Page 5: AI Insights

Purpose: summarize local Qwen interpretation and next checks.

Recommended visuals:

- Text box: AI interpretation
- Table: recommended SQL checks
- Table: recommended DAX measures

## Storytelling Flow

1. Start with how many records are in scope.
2. Explain whether the data is trustworthy enough for BI.
3. Show which WUA/segment dominates the data.
4. Compare payment collection with service usage.
5. Close with AI-supported next checks and dashboard actions.
""".strip()


def save_powerbi_automation_pack(database_name, full_table_name, df, outputs, table_targets=None):
    os.makedirs(POWERBI_OUTPUT_DIR, exist_ok=True)

    data_dictionary_df = build_powerbi_data_dictionary(df, outputs)
    risk_kpi_definitions_df = build_powerbi_risk_kpi_definitions(outputs)

    file_paths = {
        "data_dictionary": os.path.join(POWERBI_OUTPUT_DIR, "data_dictionary.csv"),
        "risk_kpi_definitions": os.path.join(POWERBI_OUTPUT_DIR, "risk_kpi_definitions.csv"),
        "measures_dax": os.path.join(POWERBI_OUTPUT_DIR, "measures.dax"),
        "power_query_scripts": os.path.join(POWERBI_OUTPUT_DIR, "power_query_import_scripts.m"),
        "sql_views": os.path.join(POWERBI_OUTPUT_DIR, "sql_views_for_powerbi.sql"),
        "dashboard_blueprint": os.path.join(POWERBI_OUTPUT_DIR, "dashboard_blueprint.md"),
        "manifest": os.path.join(POWERBI_OUTPUT_DIR, "manifest.json"),
    }

    data_dictionary_df.to_csv(file_paths["data_dictionary"], index=False, encoding="utf-8-sig")
    risk_kpi_definitions_df.to_csv(file_paths["risk_kpi_definitions"], index=False, encoding="utf-8-sig")

    with open(file_paths["measures_dax"], "w", encoding="utf-8") as file:
        file.write(build_powerbi_measures_dax(outputs))

    with open(file_paths["power_query_scripts"], "w", encoding="utf-8") as file:
        file.write(build_power_query_import_scripts())

    with open(file_paths["sql_views"], "w", encoding="utf-8") as file:
        file.write(build_powerbi_sql_views(database_name, full_table_name, outputs, table_targets))

    with open(file_paths["dashboard_blueprint"], "w", encoding="utf-8") as file:
        file.write(build_dashboard_blueprint(database_name, full_table_name, outputs, table_targets))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "server": SERVER,
        "database": database_name,
        "table": full_table_name,
        "table_targets": table_targets or [],
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "risk_score": outputs["risk_score"],
        "detected_columns": outputs["detected_columns"],
        "files": file_paths,
        "notes": [
            "All files are generated locally.",
            "Streamlit does not publish to Power BI Service.",
            "SQL scripts are generated for review and are not executed automatically.",
        ],
    }

    with open(file_paths["manifest"], "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return file_paths



# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Database Explorer")

try:
    databases = get_databases()
except Exception as e:
    st.sidebar.error("Could not connect to SQL Server.")
    st.exception(e)
    st.stop()

if not databases:
    st.sidebar.warning("No user databases were found on this SQL Server.")
    st.stop()

selected_databases = st.sidebar.multiselect(
    "Select database(s)",
    databases,
    default=[databases[0]]
)

if not selected_databases:
    st.sidebar.warning("Select at least one database.")
    st.stop()

tables_df, table_load_errors = get_tables_for_databases(selected_databases)

for database_name, error_text in table_load_errors:
    st.sidebar.warning(f"Could not load tables from {database_name}: {error_text}")

if tables_df.empty:
    st.sidebar.warning("No base tables were found in the selected database(s).")
    st.stop()

table_selection_mode = st.sidebar.radio(
    "Table selection mode",
    [
        "One exact table",
        "Several exact tables",
        "Same schema.table across selected databases",
    ],
)

if table_selection_mode == "One exact table":
    selected_table_display = st.sidebar.selectbox(
        "Select table",
        tables_df["DISPLAY_TABLE_NAME"].tolist(),
    )
    selected_table_targets_df = tables_df[
        tables_df["DISPLAY_TABLE_NAME"] == selected_table_display
    ].copy()

elif table_selection_mode == "Several exact tables":
    selected_table_displays = st.sidebar.multiselect(
        "Select table(s)",
        tables_df["DISPLAY_TABLE_NAME"].tolist(),
        default=[tables_df["DISPLAY_TABLE_NAME"].iloc[0]],
    )

    if not selected_table_displays:
        st.sidebar.warning("Select at least one table.")
        st.stop()

    selected_table_targets_df = tables_df[
        tables_df["DISPLAY_TABLE_NAME"].isin(selected_table_displays)
    ].copy()

else:
    schema_table_options = sorted(tables_df["FULL_TABLE_NAME"].unique().tolist())
    selected_schema_table = st.sidebar.selectbox(
        "Select schema.table",
        schema_table_options,
    )

    selected_table_targets_df = tables_df[
        tables_df["FULL_TABLE_NAME"] == selected_schema_table
    ].copy()

    found_databases = selected_table_targets_df["DATABASE_NAME"].tolist()
    missing_databases = [
        database_name
        for database_name in selected_databases
        if database_name not in found_databases
    ]

    st.sidebar.caption(
        f"Found `{selected_schema_table}` in {len(found_databases)} of "
        f"{len(selected_databases)} selected database(s)."
    )

    if missing_databases:
        st.sidebar.warning(
            "This table was not found in: " + ", ".join(missing_databases)
        )

selected_full_table_names = selected_table_targets_df["FULL_TABLE_NAME"].unique().tolist()
selection_is_single_table = len(selected_table_targets_df) == 1
selection_has_common_table = len(selected_full_table_names) == 1

if selection_is_single_table:
    selected_table_row = selected_table_targets_df.iloc[0]
    selected_database = selected_table_row["DATABASE_NAME"]
    selected_table = selected_table_row["FULL_TABLE_NAME"]
    selection_description = selected_table_row["DISPLAY_TABLE_NAME"]
elif selection_has_common_table:
    selected_database = "Multiple selected databases"
    selected_table = selected_full_table_names[0]
    selection_description = (
        f"{selected_table} across {len(selected_table_targets_df)} database(s)"
    )
else:
    selected_database = "Multiple"
    selected_table = "Combined table selection"
    selection_description = (
        f"{len(selected_table_targets_df)} table targets selected"
    )

top_n = st.sidebar.slider(
    "Rows to load",
    min_value=10,
    max_value=500000,
    value=100,
    step=100
)

load_button = st.sidebar.button("Load Selected Table(s)")


# -----------------------------
# LOCAL RAG KNOWLEDGE BASE
# -----------------------------

st.subheader("Local RAG Knowledge Base")
st.caption("Place .txt, .md, or .pdf files inside knowledge_base/. ChromaDB stores embeddings locally in chroma_db/.")

use_rag = False

if RAG_IMPORT_ERROR:
    st.error(f"RAG engine could not be imported: {RAG_IMPORT_ERROR}")
else:
    use_rag = st.checkbox("Use RAG Knowledge Base")

    rag_col1, rag_col2, rag_col3 = st.columns(3)

    with rag_col1:
        if st.button("Ingest knowledge base"):
            try:
                with st.spinner("Reading documents, chunking text, creating embeddings, and saving to ChromaDB..."):
                    ingest_result = ingest_documents()
                st.success(ingest_result["message"])
            except Exception as e:
                st.error("Could not ingest knowledge base.")
                st.exception(e)

    with rag_col2:
        if st.button("Clear vector DB"):
            try:
                clear_result = clear_vector_db()
                st.session_state.pop("rag_matches", None)
                st.session_state.pop("rag_context", None)
                st.success(clear_result["message"])
            except Exception as e:
                st.error("Could not clear ChromaDB.")
                st.exception(e)

    with rag_col3:
        if st.button("Show indexed documents count"):
            try:
                rag_stats = get_indexed_documents_count()
                st.info(f"Indexed documents: {rag_stats['documents']} | Chunks: {rag_stats['chunks']}")
            except Exception as e:
                st.error("Could not read ChromaDB index count.")
                st.exception(e)


# -----------------------------
# AGENT MEMORY CENTER
# -----------------------------

st.subheader("Agent Memory Center")
st.caption(
    "Local memory for solved agent tasks plus a read-only SQL Server context scan. "
    "The database context stores metadata only: databases, tables, columns, types, row-count estimates, and detected concepts. "
    "Search uses synonyms and business concepts, not only exact names."
)

use_sql_server_context_memory = False

if MEMORY_IMPORT_ERROR:
    st.error(f"Memory engine could not be imported: {MEMORY_IMPORT_ERROR}")
else:
    memory_stats = get_memory_stats()
    mem_col1, mem_col2, mem_col3, mem_col4 = st.columns(4)
    mem_col1.metric("Remembered Tasks", memory_stats["task_count"])
    mem_col2.metric("Databases in Memory", memory_stats["database_count"])
    mem_col3.metric("Tables in Memory", memory_stats["table_count"])
    mem_col4.metric("Columns in Memory", memory_stats["column_count"])

    use_sql_server_context_memory = st.checkbox(
        "Use SQL Server context memory for agent runs",
        value=bool(memory_stats["has_database_context"]),
        key="use_sql_server_context_memory",
    )

    memory_action_col1, memory_action_col2 = st.columns([1, 2])

    with memory_action_col1:
        if st.button("Refresh SQL Server Context Memory"):
            try:
                with st.spinner("Scanning all local SQL Server databases read-only. Metadata only, no raw table rows..."):
                    context = scan_sql_server_context(databases, connect_to_database)
                st.success(
                    "SQL Server context memory refreshed: "
                    f"{context['totals']['database_count']} databases, "
                    f"{context['totals']['table_count']} tables, "
                    f"{context['totals']['column_count']} columns."
                )
                st.write(f"- JSON: `{DATABASE_CONTEXT_JSON_PATH}`")
                st.write(f"- Markdown: `{DATABASE_CONTEXT_MD_PATH}`")
            except Exception as e:
                st.error("Could not refresh SQL Server context memory.")
                st.exception(e)

    with memory_action_col2:
        if memory_stats["context_generated_at"]:
            st.info(
                "Current SQL Server context memory generated at "
                f"`{memory_stats['context_generated_at']}`. "
                f"Estimated rows across scanned databases: `{memory_stats['estimated_rows']:,}`."
            )
        else:
            st.info("No SQL Server context memory yet. Click refresh to let agents learn the database landscape.")

    with st.expander("Search SQL Server context memory"):
        context_search_query = st.text_input(
            "Search tables/columns/concepts",
            value="payment contract water user wua",
            key="sql_server_context_memory_search",
        )

        context_matches = search_database_context(context_search_query, max_results=30)
        if context_matches:
            context_rows = []
            for match in context_matches:
                context_rows.append({
                    "Score": match["score"],
                    "Database": match["database"],
                    "Table": match["table"],
                    "RowsEstimate": match["estimated_rows"],
                    "Concepts": ", ".join(match["detected_concepts"]),
                    "MatchedConcepts": ", ".join(match.get("matched_concepts", [])),
                    "MatchedTerms": ", ".join(match.get("matched_terms", [])[:12]),
                    "KeyLikeColumns": ", ".join(match["key_like_columns"][:8]),
                    "ColumnsPreview": ", ".join(match["columns"][:20]),
                })
            st.dataframe(pd.DataFrame(context_rows), use_container_width=True)
        else:
            st.info("No context matches yet. Refresh SQL Server context memory first.")

    st.markdown("### Global Agent Command Center")
    st.caption(
        "Ask agents database-wide questions without selecting a database or table. "
        "Agents use SQL Server context memory and can run safe read-only SQL when generated SQL uses fully qualified table names."
    )

    auto_coordinator_option = "Auto Coordinator (choose needed agents)"

    if AGENTS_IMPORT_ERROR:
        global_agent_options = {
            auto_coordinator_option: "Coordinate SQL, operational analytics, Power BI, and validation work.",
            "Coordinator Agent": "Coordinate SQL, operational analytics, Power BI, and validation work.",
        }
    else:
        global_agent_options = {
            auto_coordinator_option: (
                "Automatically route the command to needed specialist agents, then synthesize the result."
            ),
            **{
            agent["name"]: agent["role"]
            for agent in AGENT_DEFINITIONS.values()
            },
        }
        global_agent_options["Coordinator Agent"] = (
            "Coordinate SQL Server metadata, remembered tasks, RAG context, and specialist reasoning."
        )

    global_agent_col1, global_agent_col2 = st.columns([1, 1])

    with global_agent_col1:
        global_agent_name = st.selectbox(
            "Global agent",
            list(global_agent_options.keys()),
            key="global_agent_command_agent",
        )

    with global_agent_col2:
        global_execute_sql = st.checkbox(
            "Run safe read-only SQL from global agent",
            value=True,
            key="global_agent_execute_sql",
        )

    global_use_rag = st.checkbox(
        "Use RAG context for global command",
        value=bool(use_rag),
        disabled=bool(RAG_IMPORT_ERROR),
        key="global_agent_use_rag",
    )

    global_command = st.text_area(
        "Command for global agent:",
        value=(
            "Find the payment-related tables across all databases and suggest a safe SQL Server query "
            "to count unique paying water users using PaymentContractKod."
        ),
        height=130,
        key="global_agent_command_text",
    )

    if st.button("Run Global Agent Command", key="run_global_agent_command"):
        try:
            if not memory_stats["has_database_context"]:
                st.warning("SQL Server context memory is empty. Refresh it first for best results.")

            global_sql_context = format_database_context_for_prompt(
                global_command,
                max_results=20,
                max_chars=12000,
            )
            remembered_tasks_text = format_remembered_tasks_for_prompt(
                global_command,
                limit=5,
            )
            global_rag_matches = []
            global_rag_context = "RAG context was not used for this global command."

            if global_use_rag:
                with st.spinner("Searching local RAG knowledge base for global command..."):
                    global_rag_matches = search_knowledge(global_command, top_k=4)
                global_rag_context = format_retrieved_context(global_rag_matches)

            global_specialist_reports = {}
            routed_agent_keys = []

            if (
                global_agent_name == auto_coordinator_option
                and not AGENTS_IMPORT_ERROR
            ):
                routed_agent_keys = choose_needed_global_agents(global_command)
                routing_names = [
                    AGENT_DEFINITIONS[agent_key]["name"]
                    for agent_key in routed_agent_keys
                ]
                st.info("Auto Coordinator selected: " + ", ".join(routing_names))

                route_progress = st.progress(0)
                route_status = st.empty()

                for index, agent_key in enumerate(routed_agent_keys, start=1):
                    specialist = AGENT_DEFINITIONS[agent_key]
                    route_status.info(
                        f"Running {specialist['name']} ({index}/{len(routed_agent_keys)})..."
                    )
                    specialist_prompt = build_global_agent_command_prompt(
                        agent_name=specialist["name"],
                        agent_role=specialist["role"],
                        user_command=global_command,
                        sql_server_memory_text=global_sql_context,
                        rag_context=global_rag_context,
                        remembered_tasks_text=remembered_tasks_text,
                        allow_sql_execution=global_execute_sql,
                    )
                    global_specialist_reports[agent_key] = ask_qwen(
                        specialist_prompt,
                        max_tokens=specialist["max_tokens"],
                    )
                    route_progress.progress(index / (len(routed_agent_keys) + 1))

                route_status.info("Coordinator is synthesizing specialist work...")
                coordinator_prompt = build_global_coordinator_synthesis_prompt(
                    user_command=global_command,
                    sql_server_memory_text=global_sql_context,
                    rag_context=global_rag_context,
                    remembered_tasks_text=remembered_tasks_text,
                    specialist_reports=global_specialist_reports,
                )
                global_answer = ask_qwen(coordinator_prompt, max_tokens=9000)
                route_progress.progress(1.0)
                route_status.success("Auto Coordinator completed the routed agent workflow.")
            else:
                global_prompt = build_global_agent_command_prompt(
                    agent_name=global_agent_name,
                    agent_role=global_agent_options[global_agent_name],
                    user_command=global_command,
                    sql_server_memory_text=global_sql_context,
                    rag_context=global_rag_context,
                    remembered_tasks_text=remembered_tasks_text,
                    allow_sql_execution=global_execute_sql,
                )

                with st.spinner(f"{global_agent_name} is working from SQL Server context memory..."):
                    global_answer = ask_qwen(global_prompt, max_tokens=8000)

            global_sql_queries = []
            for report_text in list(global_specialist_reports.values()) + [global_answer]:
                for sql_query in extract_all_sql_queries(report_text):
                    if sql_query not in global_sql_queries:
                        global_sql_queries.append(sql_query)

            if (
                global_execute_sql
                and not global_sql_queries
                and question_needs_sql_result(global_command)
            ):
                sql_retry_prompt = f"""
Write one runnable SQL Server SELECT query for this database-wide command.

The query will run from a connection to the master database, so use fully
qualified table names like [Structure_2025].[dbo].[Payment].

SQL Server context memory:
{global_sql_context}

User command:
{global_command}

Rules:
- Return only one SQL Server SELECT query.
- Put it inside a ```sql code block.
- Use only databases, tables, and columns from the context memory.
- Do not match only exact words from the command. Use synonyms, table concepts,
  key-like columns, and business meaning to choose the best table and columns.
- If combining databases, add source labels as string literals, for example:
  SELECT 'Structure_2024' AS SourceDatabase, ...
- Never use SourceDatabase or SourceTable as if they are SQL Server table columns.
- Keep it read-only.
- Do not include explanation outside the SQL code block.
"""
                with st.spinner("Global agent answered without SQL. Asking for SQL-only..."):
                    sql_retry_answer = ask_qwen(sql_retry_prompt, max_tokens=3000)
                retry_sql = extract_sql_query(sql_retry_answer)
                if retry_sql:
                    global_sql_queries = [retry_sql]
                    global_answer += (
                        "\n\n## SQL-Only Retry Used For Execution\n\n"
                        "```sql\n"
                        f"{retry_sql.strip()}\n"
                        "```"
                    )

            global_sql_text = format_sql_drafts(global_sql_queries)
            global_sql_safety = build_sql_safety_summary(global_sql_queries)
            global_execution_records = []
            global_result_answer = None

            if global_execute_sql and global_sql_queries:
                with st.spinner("Running global agent SQL through the read-only safety gate..."):
                    global_execution_records = execute_global_sql_queries(global_sql_queries)

                if any(record.get("result") is not None for record in global_execution_records):
                    result_prompt = build_agent_sql_result_prompt(
                        agent_name=global_agent_name,
                        user_question=global_command,
                        selected_table="No table selected",
                        selection_description="Global SQL Server memory command",
                        agent_answer=global_answer,
                        execution_records=global_execution_records,
                    )
                    with st.spinner(f"{global_agent_name} is answering from global SQL results..."):
                        global_result_answer = ask_qwen(result_prompt, max_tokens=6000)

            memory_paths = save_agent_task_memory({
                "agent_key": "global_command",
                "agent_name": global_agent_name,
                "question": global_command,
                "selection": "Global command without selected table",
                "database": "All SQL Server context memory",
                "table": "No selected table",
                "row_count": None,
                "column_count": None,
                "rag_used": bool(global_use_rag),
                "sql_server_memory_used": True,
                "sql_text": global_sql_text,
                "sql_execution_summaries": summarize_sql_execution_records(global_execution_records),
                "final_answer": global_result_answer or global_answer,
                "agent_report_excerpt": safe_preview_text(global_answer, 6000),
                "report_files": {},
                "routed_agents": routed_agent_keys,
                "specialist_report_excerpts": {
                    agent_key: safe_preview_text(report_text, 3000)
                    for agent_key, report_text in global_specialist_reports.items()
                },
            })

            st.session_state["global_agent_answer"] = global_answer
            st.session_state["global_agent_result_answer"] = global_result_answer
            st.session_state["global_agent_sql"] = global_sql_text
            st.session_state["global_agent_sql_safety"] = global_sql_safety
            st.session_state["global_agent_sql_results"] = global_execution_records
            st.session_state["global_agent_rag_matches"] = global_rag_matches
            st.session_state["global_agent_memory_paths"] = memory_paths
            st.session_state["global_agent_routed_agents"] = routed_agent_keys
            st.session_state["global_agent_specialist_reports"] = global_specialist_reports
            st.session_state["global_agent_error"] = None

        except Exception as e:
            st.session_state["global_agent_error"] = str(e)
            st.error("Could not run global agent command.")
            st.exception(e)

    if st.session_state.get("global_agent_error"):
        st.warning(f"Last global agent error: {st.session_state['global_agent_error']}")

    if st.session_state.get("global_agent_answer"):
        if st.session_state.get("global_agent_result_answer"):
            st.markdown("#### Global Agent Final Answer From SQL Results")
            st.markdown(st.session_state["global_agent_result_answer"])

        st.markdown("#### Global Agent Report")
        st.markdown(st.session_state["global_agent_answer"])

        if st.session_state.get("global_agent_routed_agents"):
            routed_names = [
                AGENT_DEFINITIONS[agent_key]["name"]
                for agent_key in st.session_state["global_agent_routed_agents"]
                if agent_key in AGENT_DEFINITIONS
            ]
            st.info("Auto Coordinator ran: " + ", ".join(routed_names))

        if st.session_state.get("global_agent_specialist_reports"):
            with st.expander("Specialist reports used by Auto Coordinator"):
                for agent_key, report_text in st.session_state["global_agent_specialist_reports"].items():
                    agent_name = AGENT_DEFINITIONS.get(agent_key, {}).get("name", agent_key)
                    st.markdown(f"##### {agent_name}")
                    st.markdown(report_text)

        if st.session_state.get("global_agent_sql"):
            st.markdown("#### Global Agent SQL Draft")
            st.code(st.session_state["global_agent_sql"], language="sql")

        if st.session_state.get("global_agent_sql_safety"):
            st.markdown("#### Global SQL Safety Check")
            st.dataframe(pd.DataFrame(st.session_state["global_agent_sql_safety"]), use_container_width=True)

        if st.session_state.get("global_agent_sql_results"):
            st.markdown("#### Global SQL Execution Results")
            for record in st.session_state["global_agent_sql_results"]:
                with st.expander(f"Query {record['query_index']} - {record['status']}", expanded=record.get("result") is not None):
                    st.code(record["sql"], language="sql")
                    if record.get("error"):
                        st.warning(record["error"])
                    if record.get("result") is not None:
                        result_df = record["result"]
                        st.write(f"Rows returned: **{len(result_df)}**")
                        show_dataframe_preview(
                            result_df,
                            max_rows=QUERY_RESULT_PREVIEW_ROWS,
                            context_label=f"global agent query {record['query_index']} result",
                        )
                        show_auto_result_chart(result_df)

        if st.session_state.get("global_agent_memory_paths"):
            st.caption(
                "Saved to memory: "
                f"`{st.session_state['global_agent_memory_paths']['latest_task']}`"
            )

    with st.expander("Search remembered agent tasks"):
        task_search_query = st.text_input(
            "Search solved tasks",
            value="",
            key="agent_task_memory_search",
        )
        remembered_tasks = load_agent_task_history(search_text=task_search_query, limit=30)

        if remembered_tasks:
            task_rows = []
            for task in remembered_tasks:
                task_rows.append({
                    "SavedAt": task.get("saved_at"),
                    "Agent": task.get("agent_name"),
                    "Selection": task.get("selection"),
                    "Question": safe_preview_text(task.get("question"), 160),
                    "FinalAnswer": safe_preview_text(task.get("final_answer"), 220),
                    "SQLQueries": len(task.get("sql_execution_summaries") or []),
                })
            st.dataframe(pd.DataFrame(task_rows), use_container_width=True)

            selected_task_index = st.number_input(
                "Open remembered task number",
                min_value=1,
                max_value=len(remembered_tasks),
                value=1,
                step=1,
            )
            selected_task = remembered_tasks[int(selected_task_index) - 1]
            st.json(selected_task)
        else:
            st.info("No remembered tasks found yet. Run an agent with SQL execution to create memory.")


# -----------------------------
# ADVANCED BI TRAINING TOOLS
# Hidden by default so the main demo stays focused on the local SQL AI assistant.
# -----------------------------

if SHOW_ADVANCED_BI_TRAINING:
    # This optional export/training layer is kept for future demos and Power BI practice.
    # -----------------------------
    # OPERATIONAL BI TRAINING
    # -----------------------------
    
    st.subheader("Operational BI Training")
    st.caption(
        "Use irrigation/WUA data as a local training dataset for business-style BI, "
        "segment monitoring, and data quality thinking."
    )
    st.info(
        "Power BI can connect directly to SQL Server using Get Data -> SQL Server, with Import or DirectQuery mode. "
        "For this learning demo, exported CSVs are also provided for simple Power BI practice."
    )
    
    risk_source_mode = st.radio(
        "BI training source mode",
        [
            "One database table",
            "Multiple exact tables",
            "Same schema.table across multiple databases",
        ],
        horizontal=True,
    )
    
    risk_source_ready = False
    risk_targets_df = pd.DataFrame()
    risk_database_label = ""
    risk_table_label = ""
    
    if risk_source_mode == "One database table":
        risk_db_col, risk_table_col, risk_limit_col = st.columns(3)
    
        with risk_db_col:
            risk_database = st.selectbox(
                "BI training database",
                databases,
                key="risk_training_database_single",
            )
    
        try:
            risk_tables_df = get_tables(risk_database)
        except Exception as e:
            st.warning(f"Could not load tables for Operational BI Training: {e}")
            risk_tables_df = pd.DataFrame(columns=["FULL_TABLE_NAME"])
    
        if risk_tables_df.empty:
            st.info("No tables are available for the selected Operational BI database.")
        else:
            with risk_table_col:
                risk_table = st.selectbox(
                    "BI training table",
                    risk_tables_df["FULL_TABLE_NAME"].tolist(),
                    key="risk_training_table_single",
                )
    
            risk_targets_df = risk_tables_df[
                risk_tables_df["FULL_TABLE_NAME"] == risk_table
            ].copy()
            risk_targets_df.insert(0, "DATABASE_NAME", risk_database)
            risk_targets_df["DISPLAY_TABLE_NAME"] = (
                risk_database + "." + risk_targets_df["FULL_TABLE_NAME"]
            )
            risk_database_label = risk_database
            risk_table_label = risk_table
            risk_source_ready = True
    
            with risk_limit_col:
                risk_row_limit = st.slider(
                    "BI training row limit",
                    min_value=10,
                    max_value=500000,
                    value=50000,
                    step=1000,
                    key="risk_training_row_limit_single",
                )
    
    elif risk_source_mode == "Multiple exact tables":
        risk_db_col, risk_table_col, risk_limit_col = st.columns(3)
    
        with risk_db_col:
            default_risk_databases = databases[: min(2, len(databases))]
            risk_databases = st.multiselect(
                "BI training databases",
                databases,
                default=default_risk_databases,
                key="risk_training_databases_exact_tables",
            )
    
        if not risk_databases:
            st.info("Select at least one database for multiple-table Operational BI export.")
        else:
            risk_all_tables_df, risk_table_errors = get_tables_for_databases(risk_databases)
    
            for database_name, error_text in risk_table_errors:
                st.warning(f"Could not load tables from {database_name}: {error_text}")
    
            if risk_all_tables_df.empty:
                st.info("No tables were found in the selected Operational BI databases.")
            else:
                with risk_table_col:
                    table_options = risk_all_tables_df["DISPLAY_TABLE_NAME"].tolist()
                    default_tables = table_options[: min(2, len(table_options))]
                    risk_table_displays = st.multiselect(
                        "BI training table(s)",
                        table_options,
                        default=default_tables,
                        key="risk_training_tables_exact",
                    )
    
                if not risk_table_displays:
                    st.info("Select at least one table for the Operational BI export.")
                else:
                    risk_targets_df = risk_all_tables_df[
                        risk_all_tables_df["DISPLAY_TABLE_NAME"].isin(risk_table_displays)
                    ].copy()
    
                    risk_database_label = "Multiple selected databases/tables"
                    risk_table_label = f"{len(risk_targets_df)} exact table target(s)"
                    risk_source_ready = len(risk_targets_df) > 0
    
                    with risk_limit_col:
                        risk_row_limit = st.slider(
                            "Rows per table target",
                            min_value=10,
                            max_value=500000,
                            value=50000,
                            step=1000,
                            key="risk_training_row_limit_exact_tables",
                        )
    
    else:
        risk_db_col, risk_table_col, risk_limit_col = st.columns(3)
    
        with risk_db_col:
            default_risk_databases = databases[: min(2, len(databases))]
            risk_databases = st.multiselect(
                "BI training databases",
                databases,
                default=default_risk_databases,
                key="risk_training_databases_multi",
            )
    
        if not risk_databases:
            st.info("Select at least one database for the multi-database Operational BI export.")
        else:
            risk_all_tables_df, risk_table_errors = get_tables_for_databases(risk_databases)
    
            for database_name, error_text in risk_table_errors:
                st.warning(f"Could not load tables from {database_name}: {error_text}")
    
            if risk_all_tables_df.empty:
                st.info("No matching tables were found in the selected Operational BI databases.")
            else:
                with risk_table_col:
                    risk_schema_table_options = sorted(
                        risk_all_tables_df["FULL_TABLE_NAME"].unique().tolist()
                    )
                    risk_table = st.selectbox(
                        "Shared schema.table",
                        risk_schema_table_options,
                        key="risk_training_table_multi",
                    )
    
                risk_targets_df = risk_all_tables_df[
                    risk_all_tables_df["FULL_TABLE_NAME"] == risk_table
                ].copy()
    
                found_risk_databases = risk_targets_df["DATABASE_NAME"].tolist()
                missing_risk_databases = [
                    database_name
                    for database_name in risk_databases
                    if database_name not in found_risk_databases
                ]
    
                st.caption(
                    f"Found `{risk_table}` in {len(found_risk_databases)} of "
                    f"{len(risk_databases)} selected database(s)."
                )
    
                if missing_risk_databases:
                    st.warning(
                        "This table was not found in: " + ", ".join(missing_risk_databases)
                    )
    
                risk_database_label = "Multiple: " + ", ".join(found_risk_databases)
                risk_table_label = risk_table
                risk_source_ready = len(risk_targets_df) > 0
    
                with risk_limit_col:
                    risk_row_limit = st.slider(
                        "Rows per database",
                        min_value=10,
                        max_value=500000,
                        value=50000,
                        step=1000,
                        key="risk_training_row_limit_multi",
                    )
    
    if risk_source_ready and st.button("Generate Power BI-ready BI outputs"):
        try:
            with st.spinner("Loading table target(s) read-only and building Power BI CSV outputs..."):
                risk_df, risk_load_errors = load_selected_table_targets(
                    risk_targets_df,
                    risk_row_limit,
                )
    
                if risk_load_errors:
                    for display_name, error_text in risk_load_errors:
                        st.warning(f"Could not load {display_name}: {error_text}")
    
                if risk_df.empty and risk_load_errors:
                    raise ValueError("No selected Operational BI table target could be loaded.")
    
                risk_outputs = build_risk_mis_outputs(risk_df)
                risk_file_paths = save_powerbi_outputs(risk_outputs)
    
            st.session_state["risk_training_df"] = risk_df
            st.session_state["risk_training_outputs"] = risk_outputs
            st.session_state["risk_training_file_paths"] = risk_file_paths
            st.session_state["risk_training_selected_database"] = risk_database_label
            st.session_state["risk_training_selected_table"] = risk_table_label
            st.session_state["risk_training_table_targets"] = risk_targets_df[
                ["DATABASE_NAME", "FULL_TABLE_NAME", "DISPLAY_TABLE_NAME"]
            ].to_dict("records")
            st.session_state["risk_training_error"] = None
            st.session_state.pop("risk_training_ai_answer", None)
            st.session_state.pop("powerbi_automation_pack_paths", None)
            st.success(f"Power BI-ready CSV files saved to `{POWERBI_OUTPUT_DIR}`.")
        except Exception as e:
            st.session_state["risk_training_error"] = str(e)
            st.error("Could not generate Operational BI outputs.")
            st.exception(e)
    
    if st.session_state.get("risk_training_error"):
        st.warning(st.session_state["risk_training_error"])
    
    if st.session_state.get("risk_training_outputs"):
        risk_outputs = st.session_state["risk_training_outputs"]
        risk_df = st.session_state["risk_training_df"]
        risk_file_paths = st.session_state["risk_training_file_paths"]
        risk_training_database = st.session_state["risk_training_selected_database"]
        risk_training_table = st.session_state["risk_training_selected_table"]
        risk_training_table_targets = st.session_state.get("risk_training_table_targets", [])
    
        st.markdown("### BI Training Output Preview")
    
        risk_metric_col1, risk_metric_col2, risk_metric_col3, risk_metric_col4 = st.columns(4)
        risk_metric_col1.metric("Training Rows", len(risk_df))
        risk_metric_col2.metric("Columns", len(risk_df.columns))
        risk_metric_col3.metric("Readiness Score", risk_outputs["risk_score"])
        risk_metric_col4.metric("Export Files", len(risk_file_paths))
    
        st.write("**Detected training columns:**")
        st.json(risk_outputs["detected_columns"])
    
        st.write("**Saved CSV outputs:**")
        for label, file_path in risk_file_paths.items():
            st.write(f"- `{label}`: `{file_path}`")
    
        if len(risk_training_table_targets) > 1:
            st.write("**Loaded database/table targets:**")
            show_dataframe_preview(
                pd.DataFrame(risk_training_table_targets),
                max_rows=BROWSER_PREVIEW_ROWS,
                context_label="BI table target",
            )
    
        if st.button("Generate Power BI Automation Pack"):
            try:
                with st.spinner("Generating DAX, Power Query M, SQL views, dictionary, and dashboard blueprint..."):
                    powerbi_pack_paths = save_powerbi_automation_pack(
                        database_name=risk_training_database,
                        full_table_name=risk_training_table,
                        df=risk_df,
                        outputs=risk_outputs,
                        table_targets=risk_training_table_targets,
                    )
    
                st.session_state["powerbi_automation_pack_paths"] = powerbi_pack_paths
                st.success(f"Power BI Automation Pack saved to `{POWERBI_OUTPUT_DIR}`.")
            except Exception as e:
                st.error("Could not generate Power BI Automation Pack.")
                st.exception(e)
    
        if st.session_state.get("powerbi_automation_pack_paths"):
            st.markdown("### Power BI Automation Pack Files")
            pack_paths = st.session_state["powerbi_automation_pack_paths"]
            for label, file_path in pack_paths.items():
                st.write(f"- `{label}`: `{file_path}`")
    
        risk_tab1, risk_tab2, risk_tab3, risk_tab4 = st.tabs([
            "KPI Summary",
            "Data Quality",
            "Analytics Summary",
            "Clean Export Preview",
        ])
    
        with risk_tab1:
            show_dataframe_preview(
                risk_outputs["kpi_summary_df"],
                max_rows=BROWSER_PREVIEW_ROWS,
                context_label="KPI summary",
            )
    
        with risk_tab2:
            show_dataframe_preview(
                risk_outputs["data_quality_df"],
                max_rows=BROWSER_PREVIEW_ROWS,
                context_label="data quality summary",
            )
    
        with risk_tab3:
            show_dataframe_preview(
                risk_outputs["risk_training_summary_df"],
                max_rows=BROWSER_PREVIEW_ROWS,
                context_label="BI training summary",
            )
    
        with risk_tab4:
            show_dataframe_preview(
                risk_outputs["clean_df"],
                max_rows=BROWSER_PREVIEW_ROWS,
                context_label="clean export",
            )
    
        st.markdown("### Power BI Guidance")
        st.markdown(
            """
    Recommended Power BI pages:
    - Page 1: Executive Overview
    - Page 2: Data Quality Readiness
    - Page 3: WUA / Segment Concentration
    - Page 4: Payment vs Water Delivery
    - Page 5: AI Insights
    
    Recommended visuals:
    - KPI cards for total records, readiness score, duplicate rows, missing contact rate
    - Bar chart for missing values by column
    - Line chart for registration trend or monthly payments
    - Treemap or bar chart for WUA / segment concentration
    - Scatter chart for water delivered vs payment collected when both fields exist
    - Table visual for high-issue records or columns
    
    Dashboard storytelling:
    - Start with dataset size and data readiness
    - Explain where data quality issues are concentrated
    - Show whether one WUA/segment dominates the data
    - Compare payment collection with service usage
    - End with AI-generated observations and next checks
    """
        )
    
        st.code(
            """
    -- Example SQL pattern for monthly payment trend
    SELECT
        DATEFROMPARTS(YEAR([PaymentDate]), MONTH([PaymentDate]), 1) AS PaymentMonth,
        SUM(CAST([PaymentMoney] AS float)) AS TotalPayments
    FROM [dbo].[Payment]
    WHERE [PaymentDate] IS NOT NULL
      AND [PaymentMoney] IS NOT NULL
    GROUP BY DATEFROMPARTS(YEAR([PaymentDate]), MONTH([PaymentDate]), 1)
    ORDER BY PaymentMonth;
    """.strip(),
            language="sql",
        )
    
        st.code(
            """
    Total Records = COUNTROWS(clean_table_export)
    
    Missing Values = SUM(data_quality_summary[MissingValues])
    
    High Missing Columns =
    COUNTROWS(
        FILTER(data_quality_summary, data_quality_summary[MissingRatePct] >= 30)
    )
    
    Average Missing Rate = AVERAGE(data_quality_summary[MissingRatePct])
    
    Readiness Score = MAX(risk_training_summary[Value])
    """.strip(),
            language="text",
        )
    
        risk_ai_question = st.text_area(
            "Question for Operational BI interpretation:",
            value=(
                "Interpret this table as an Operational BI analyst. Explain key data quality issues, "
                "segment concentration, possible payment/collection gaps, and what Power BI dashboard "
                "story should be built from it."
            ),
            height=130,
            key="risk_training_ai_question",
        )
    
        if st.button("Generate Operational BI Interpretation"):
            try:
                rag_context = "RAG knowledge base is disabled."
    
                if use_rag:
                    with st.spinner("Searching local RAG knowledge base for analytics/BI context..."):
                        rag_matches = search_knowledge(risk_ai_question, top_k=4)
                    rag_context = format_retrieved_context(rag_matches)
    
                risk_prompt = f"""
    You are a Operational BI analyst using irrigation data as a training dataset.
    Use business-style operations analytics thinking, but explain it in beginner-friendly language.
    
    Business analogy:
    - Water users = service users / customers
    - WUA = organization unit / segment / group
    - Water delivered = exposure / service usage
    - Payment collected = payment collection
    - Unpaid amount / debt = unpaid service amount
    - Missing passport/phone/address = data quality issue
    - Abnormal water delivery vs low payment = operational anomaly
    
    Selected database:
    {risk_training_database}
    
    Selected table:
    {risk_training_table}
    
    Selected table targets:
    {pd.DataFrame(risk_training_table_targets).to_string(index=False) if risk_training_table_targets else "One selected table target."}
    
    Columns:
    {risk_df.columns.tolist()}
    
    KPI summary:
    {risk_outputs["kpi_summary_df"].to_string(index=False)}
    
    Data quality summary:
    {risk_outputs["data_quality_df"].head(40).to_string(index=False)}
    
    Operational analytics summary:
    {risk_outputs["risk_training_summary_df"].head(80).to_string(index=False)}
    
    Retrieved RAG context:
    {rag_context}
    
    User question:
    {risk_ai_question}
    
    Please answer as:
    - executive summary
    - key BI/data quality observations
    - data quality observations
    - segment concentration observations
    - payment/service usage observations if columns exist
    - recommended Power BI pages and visuals
    - recommended DAX measures
    - recommended SQL checks
    - practical next steps
    """
    
                with st.spinner("Asking local Qwen for Operational BI interpretation..."):
                    risk_ai_answer = ask_qwen(risk_prompt, max_tokens=6000)
    
                st.session_state["risk_training_ai_answer"] = risk_ai_answer
            except Exception as e:
                st.error("Could not generate Operational BI interpretation.")
                st.exception(e)
    
    if st.session_state.get("risk_training_ai_answer"):
        st.markdown("### Operational BI AI Interpretation")
        st.markdown(st.session_state["risk_training_ai_answer"])
    
    

# -----------------------------
# LOAD DATA
# -----------------------------

if load_button:
    try:
        df, load_errors = load_selected_table_targets(
            selected_table_targets_df,
            top_n,
        )

        if load_errors:
            for display_name, error_text in load_errors:
                st.warning(f"Could not load {display_name}: {error_text}")

        if df.empty and load_errors:
            raise ValueError("No selected table targets could be loaded.")

        st.session_state["df"] = df
        st.session_state["selected_database"] = selected_database
        st.session_state["selected_table"] = selected_table
        st.session_state["selection_description"] = selection_description
        st.session_state["selection_is_single_table"] = selection_is_single_table
        st.session_state["selection_has_common_table"] = selection_has_common_table
        st.session_state["selected_table_targets"] = selected_table_targets_df[
            ["DATABASE_NAME", "FULL_TABLE_NAME", "DISPLAY_TABLE_NAME"]
        ].to_dict("records")
        st.session_state.pop("qwen_answer", None)
        st.session_state.pop("qwen_question", None)
        st.session_state.pop("qwen_context_note", None)
        st.session_state.pop("qwen_error", None)
        st.session_state.pop("qwen_sql", None)
        st.session_state.pop("qwen_original_sql", None)
        st.session_state.pop("qwen_sql_repaired", None)
        st.session_state.pop("qwen_sql_fallback_used", None)
        st.session_state.pop("qwen_sql_repair_answer", None)
        st.session_state.pop("qwen_sql_only_answer", None)
        st.session_state.pop("qwen_sql_result", None)
        st.session_state.pop("qwen_sql_error", None)
        st.session_state.pop("qwen_sql_target_errors", None)
        st.session_state.pop("rag_matches", None)
        st.session_state.pop("rag_context", None)
        st.session_state.pop("rag_error", None)
        st.session_state.pop("rag_enabled", None)
        st.session_state.pop("full_sql_chart_query", None)
        st.session_state.pop("full_sql_chart_df", None)
        st.session_state.pop("full_sql_chart_error", None)
        st.session_state.pop("agent_workspace_answer", None)
        st.session_state.pop("agent_workspace_agent_key", None)
        st.session_state.pop("agent_workspace_agent_name", None)
        st.session_state.pop("agent_workspace_sql", None)
        st.session_state.pop("agent_workspace_sql_safety", None)
        st.session_state.pop("agent_workspace_sql_results", None)
        st.session_state.pop("agent_workspace_result_answer", None)
        st.session_state.pop("agent_workspace_files", None)
        st.session_state.pop("agent_workspace_rag_matches", None)
        st.session_state.pop("agent_workspace_team_reports", None)
        st.session_state.pop("agent_workspace_error", None)
        st.success("Table selection loaded successfully.")
    except Exception as e:
        st.error("Could not load selected table selection.")
        st.exception(e)

if "df" not in st.session_state:
    st.info("Select database(s) and table(s) from the sidebar, then click **Load Selected Table(s)**.")
    st.stop()

df = st.session_state["df"]
selected_database = st.session_state["selected_database"]
selected_table = st.session_state["selected_table"]
selection_description = st.session_state.get("selection_description", selected_table)
selection_is_single_table = st.session_state.get("selection_is_single_table", True)
selection_has_common_table = st.session_state.get("selection_has_common_table", selection_is_single_table)
selected_table_targets = st.session_state.get("selected_table_targets", [])


# -----------------------------
# TABLE PREVIEW
# -----------------------------

st.subheader("Selected Data")

st.write(f"**Selection:** `{selection_description}`")

if selection_is_single_table:
    st.write(f"**Database:** `{selected_database}`")
    st.write(f"**Table:** `{selected_table}`")
else:
    st.write(f"**Loaded table targets:** `{len(selected_table_targets)}`")
    with st.expander("Loaded targets"):
        show_dataframe_preview(
            pd.DataFrame(selected_table_targets),
            max_rows=BROWSER_PREVIEW_ROWS,
            context_label="target",
        )

show_dataframe_preview(
    df,
    max_rows=BROWSER_PREVIEW_ROWS,
    context_label="loaded",
)

if df.empty:
    st.warning("This table loaded successfully, but it returned 0 rows for the selected TOP N value.")


# -----------------------------
# BASIC KPIs
# -----------------------------

st.subheader("Automatic Data KPIs")

total_rows = len(df)
column_count = len(df.columns)
missing_values_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows Loaded", total_rows)
col2.metric("Columns", column_count)
col3.metric("Missing Values", missing_values_total)
col4.metric("Duplicate Rows", duplicate_rows)


# -----------------------------
# COLUMN INFO
# -----------------------------

st.subheader("Column Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Missing Values": df.isna().sum().values,
    "Missing %": (
        ((df.isna().sum() / len(df)) * 100).round(2).values
        if len(df) > 0
        else [0] * len(df.columns)
    )
})

show_dataframe_preview(
    dtype_df,
    max_rows=BROWSER_PREVIEW_ROWS,
    context_label="column info",
)


# -----------------------------
# MISSING VALUES CHART
# -----------------------------

st.subheader("Missing Values by Column")

missing_by_column = df.isna().sum()
missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

if df.empty:
    st.info("No missing value chart is available because the loaded table has 0 rows.")
elif len(missing_by_column) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_by_column.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    st.pyplot(fig)
else:
    st.success("No missing values found in loaded rows.")


# -----------------------------
# SIMPLE VALUE COUNTS CHART
# -----------------------------

st.subheader("Categorical Column Distribution")

categorical_columns = df.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

if df.empty:
    st.info("No categorical distribution chart is available because the loaded table has 0 rows.")
elif categorical_columns:
    selected_cat_col = st.selectbox(
        "Select categorical column",
        categorical_columns
    )

    value_counts = df[selected_cat_col].value_counts(dropna=False).head(20)

    if len(value_counts) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        value_counts.plot(kind="bar", ax=ax2)
        ax2.set_title(f"Top Values in {selected_cat_col}")
        ax2.set_xlabel(selected_cat_col)
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    else:
        st.info(f"No values found to plot for `{selected_cat_col}`.")
else:
    st.info("No categorical columns detected.")


# -----------------------------
# FULL SQL TABLE CHART
# -----------------------------

st.subheader("Full SQL Table Chart")
st.caption(
    "This aggregates directly in SQL Server across the full selected table, "
    "not just the rows shown in the preview."
)

date_candidates, numeric_candidates = get_full_chart_column_candidates(df)

if not selection_is_single_table:
    st.info("Full SQL table chart is available when exactly one table target is loaded.")
elif df.empty:
    st.info("Load rows from the table first so the app can detect chart columns.")
elif not date_candidates:
    st.info("No date/time column was detected for a full-table trend chart.")
else:
    chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)

    with chart_col1:
        full_chart_date_column = st.selectbox(
            "Date/time column",
            date_candidates,
            key="full_chart_date_column",
        )

    aggregation_options = ["SUM", "AVG", "COUNT rows", "MIN", "MAX"] if numeric_candidates else ["COUNT rows"]

    with chart_col2:
        full_chart_aggregation = st.selectbox(
            "Aggregation",
            aggregation_options,
            key="full_chart_aggregation",
        )

    full_chart_value_column = None
    with chart_col3:
        if full_chart_aggregation != "COUNT rows":
            full_chart_value_column = st.selectbox(
                "Value column",
                numeric_candidates,
                key="full_chart_value_column",
            )
        else:
            st.write("Value column")
            st.info("Counting rows")

    with chart_col4:
        full_chart_time_grain = st.selectbox(
            "Time grain",
            ["Month", "Day", "Year"],
            key="full_chart_time_grain",
        )

    if st.button("Build full-table chart"):
        try:
            full_chart_query = build_time_aggregate_sql(
                selected_table=selected_table,
                date_column=full_chart_date_column,
                value_column=full_chart_value_column,
                aggregation=full_chart_aggregation,
                time_grain=full_chart_time_grain,
            )

            with st.spinner("SQL Server is aggregating the full selected table..."):
                full_chart_df = run_read_only_query(selected_database, full_chart_query)

            st.session_state["full_sql_chart_query"] = full_chart_query
            st.session_state["full_sql_chart_df"] = full_chart_df
            st.session_state.pop("full_sql_chart_error", None)
        except Exception as e:
            st.session_state["full_sql_chart_error"] = str(e)
            st.session_state.pop("full_sql_chart_df", None)

    if st.session_state.get("full_sql_chart_query"):
        st.markdown("### Full-Table Chart SQL")
        st.code(st.session_state["full_sql_chart_query"], language="sql")

    if st.session_state.get("full_sql_chart_error"):
        st.warning(f"Could not build the full-table chart: {st.session_state['full_sql_chart_error']}")

    if st.session_state.get("full_sql_chart_df") is not None:
        full_chart_df = st.session_state["full_sql_chart_df"]
        st.write(f"Aggregated rows returned: **{len(full_chart_df)}**")
        show_dataframe_preview(
            full_chart_df,
            max_rows=QUERY_RESULT_PREVIEW_ROWS,
            context_label="aggregated result",
        )
        show_auto_result_chart(full_chart_df)


# -----------------------------
# AI AGENT WORKSPACE
# -----------------------------

st.subheader("AI Agent Workspace")
st.caption(
    "Run safe local specialist agents against the currently loaded dataframe. "
    "Agents can explain, recommend, and write read-only SQL, but Python still controls execution."
)

if AGENTS_IMPORT_ERROR:
    st.error(f"Agent module could not be imported: {AGENTS_IMPORT_ERROR}")
else:
    agent_question = st.text_area(
        "Question or task for the agents:",
        value=(
            "Review this selected data as a local analytics training project. "
            "Explain what is important, what risks exist, what SQL/Power BI work should come next, "
            "and what I should learn from it."
        ),
        height=120,
        key="agent_workspace_question",
    )

    agent_use_rag = st.checkbox(
        "Use RAG context for agent runs",
        value=bool(use_rag),
        disabled=bool(RAG_IMPORT_ERROR),
        key="agent_workspace_use_rag",
    )

    agent_can_execute_sql = selection_is_single_table or selection_has_common_table
    agent_execute_sql = st.checkbox(
        "Let agents run safe read-only SQL and answer from the results",
        value=bool(agent_can_execute_sql),
        disabled=not agent_can_execute_sql,
        key="agent_workspace_execute_sql",
    )

    if not agent_can_execute_sql:
        st.caption(
            "Agent SQL execution needs one exact table, or the same schema.table selected across several databases."
        )

    agent_use_sql_memory = (
        not MEMORY_IMPORT_ERROR
        and bool(st.session_state.get("use_sql_server_context_memory", False))
    )

    if agent_use_sql_memory:
        st.caption("SQL Server context memory is enabled for this agent run.")

    agent_button_order = [
        "senior_analytics_lead",
        "data_profiler",
        "data_modeler",
        "sql_analyst",
        "anomaly_detection",
        "risk_mis",
        "powerbi",
        "rag_knowledge",
        "validator",
        "full_review",
    ]

    selected_agent_key = None
    button_columns = st.columns(4)

    st.markdown("#### Specialist Agents")
    for index, agent_key in enumerate(agent_button_order):
        agent = AGENT_DEFINITIONS[agent_key]
        with button_columns[index % len(button_columns)]:
            if st.button(agent["button"], key=f"run_agent_{agent_key}"):
                selected_agent_key = agent_key

    st.markdown("#### Agent Team")
    team_col1, team_col2 = st.columns([1, 3])
    with team_col1:
        run_agent_team = st.button(
            "Run Agent Team Review",
            key="run_agent_team_review",
            help="Runs several specialist agents, then asks a coordinator agent to synthesize them.",
        )
    with team_col2:
        st.caption(
            "Team review runs Data Profiler, SQL Analyst, Operations Analytics, Power BI, Validator, "
            "and RAG Knowledge when RAG is enabled. It saves a timestamped report history."
        )

    if selected_agent_key:
        try:
            agent_evidence = build_agent_workspace_evidence(
                df=df,
                dtype_df=dtype_df,
                selected_database=selected_database,
                selected_table=selected_table,
                selection_description=selection_description,
                selected_table_targets=selected_table_targets,
            )

            if agent_use_sql_memory:
                agent_evidence["sql_server_memory_text"] = format_database_context_for_prompt(
                    agent_question,
                    max_results=12,
                )

            agent_rag_matches = []
            agent_rag_context = "RAG context was not used for this agent run."

            if agent_use_rag:
                with st.spinner("Searching local RAG knowledge base for the agent..."):
                    agent_rag_matches = search_knowledge(agent_question, top_k=4)
                agent_rag_context = format_retrieved_context(agent_rag_matches)

            agent_prompt = build_agent_prompt(
                agent_key=selected_agent_key,
                evidence=agent_evidence,
                user_question=agent_question,
                rag_context=agent_rag_context,
            )

            selected_agent = AGENT_DEFINITIONS[selected_agent_key]
            with st.spinner(f"{selected_agent['name']} is thinking locally with Qwen..."):
                agent_answer = ask_qwen(
                    agent_prompt,
                    max_tokens=selected_agent["max_tokens"],
                )

            agent_sql_queries = extract_all_sql_queries(agent_answer)

            if (
                agent_execute_sql
                and not agent_sql_queries
                and question_needs_sql_result(agent_question)
            ):
                with st.spinner("Agent answered without SQL. Asking Qwen for SQL-only so the agent can finish the task..."):
                    retry_sql_query, retry_sql_answer = ask_qwen_for_sql_only(
                        user_question=agent_question,
                        selected_table=selected_table,
                        dtype_df=dtype_df,
                        rag_context=agent_rag_context,
                    )

                if retry_sql_query:
                    agent_sql_queries = [retry_sql_query]
                    agent_answer += (
                        "\n\n## SQL-Only Retry Used For Execution\n\n"
                        "The first agent answer did not include runnable SQL, so Streamlit asked Qwen for one safe SQL draft:\n\n"
                        "```sql\n"
                        f"{retry_sql_query.strip()}\n"
                        "```"
                    )
                elif retry_sql_answer:
                    agent_answer += (
                        "\n\n## SQL-Only Retry Could Not Produce Runnable SQL\n\n"
                        f"{retry_sql_answer}"
                    )

            agent_sql = format_sql_drafts(agent_sql_queries)
            agent_sql_safety = build_sql_safety_summary(agent_sql_queries)
            agent_sql_execution_records = []
            agent_result_answer = None

            if agent_execute_sql and agent_sql_queries:
                with st.spinner("Running agent SQL drafts through the read-only safety gate..."):
                    agent_sql_execution_records = execute_agent_sql_queries(
                        sql_queries=agent_sql_queries,
                        sql_execution_enabled=agent_can_execute_sql,
                        multi_database_sql_enabled=selection_has_common_table,
                        selected_database=selected_database,
                        selected_table_targets=selected_table_targets,
                    )

                if any(record.get("result") is not None for record in agent_sql_execution_records):
                    result_prompt = build_agent_sql_result_prompt(
                        agent_name=selected_agent["name"],
                        user_question=agent_question,
                        selected_table=selected_table,
                        selection_description=selection_description,
                        agent_answer=agent_answer,
                        execution_records=agent_sql_execution_records,
                    )
                    with st.spinner(f"{selected_agent['name']} is answering from actual SQL results..."):
                        agent_result_answer = ask_qwen(result_prompt, max_tokens=5000)

            agent_metadata = {
                "selection": selection_description,
                "database": selected_database,
                "table": selected_table,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "agent_question": agent_question,
                "rag_used": bool(agent_use_rag),
                "agent_sql_execution_enabled": bool(agent_execute_sql),
            }

            report_text_to_save = agent_answer
            if agent_result_answer:
                report_text_to_save += (
                    "\n\n---\n\n"
                    "## Final Answer From Executed SQL\n\n"
                    f"{agent_result_answer}"
                )

            agent_file_paths = save_agent_report(
                agent_key=selected_agent_key,
                report_text=report_text_to_save,
                metadata=agent_metadata,
                sql_text=agent_sql,
            )

            if not MEMORY_IMPORT_ERROR:
                agent_memory_paths = save_agent_task_memory({
                    "agent_key": selected_agent_key,
                    "agent_name": selected_agent["name"],
                    "question": agent_question,
                    "selection": selection_description,
                    "database": selected_database,
                    "table": selected_table,
                    "row_count": int(len(df)),
                    "column_count": int(len(df.columns)),
                    "rag_used": bool(agent_use_rag),
                    "sql_server_memory_used": bool(agent_use_sql_memory),
                    "sql_text": agent_sql,
                    "sql_execution_summaries": summarize_sql_execution_records(agent_sql_execution_records),
                    "final_answer": agent_result_answer,
                    "agent_report_excerpt": safe_preview_text(agent_answer, 6000),
                    "report_files": agent_file_paths,
                })
                agent_file_paths.update({
                    "memory_task_history": agent_memory_paths["task_history"],
                    "memory_latest_task": agent_memory_paths["latest_task"],
                })

            st.session_state["agent_workspace_answer"] = agent_answer
            st.session_state["agent_workspace_result_answer"] = agent_result_answer
            st.session_state["agent_workspace_agent_key"] = selected_agent_key
            st.session_state["agent_workspace_agent_name"] = selected_agent["name"]
            st.session_state["agent_workspace_sql"] = agent_sql
            st.session_state["agent_workspace_sql_safety"] = agent_sql_safety
            st.session_state["agent_workspace_sql_results"] = agent_sql_execution_records
            st.session_state["agent_workspace_files"] = agent_file_paths
            st.session_state["agent_workspace_rag_matches"] = agent_rag_matches
            st.session_state["agent_workspace_team_reports"] = None
            st.session_state["agent_workspace_error"] = None

        except Exception as e:
            st.session_state["agent_workspace_error"] = str(e)
            st.error("Could not run the selected agent.")
            st.exception(e)

    if run_agent_team:
        try:
            agent_evidence = build_agent_workspace_evidence(
                df=df,
                dtype_df=dtype_df,
                selected_database=selected_database,
                selected_table=selected_table,
                selection_description=selection_description,
                selected_table_targets=selected_table_targets,
            )

            if agent_use_sql_memory:
                agent_evidence["sql_server_memory_text"] = format_database_context_for_prompt(
                    agent_question,
                    max_results=16,
                )

            agent_rag_matches = []
            agent_rag_context = "RAG context was not used for this agent team run."

            if agent_use_rag:
                with st.spinner("Searching local RAG knowledge base for the agent team..."):
                    agent_rag_matches = search_knowledge(agent_question, top_k=6)
                agent_rag_context = format_retrieved_context(agent_rag_matches)

            team_order = list(AGENT_TEAM_ORDER)
            if agent_use_rag and "rag_knowledge" not in team_order:
                team_order.insert(max(len(team_order) - 1, 0), "rag_knowledge")

            team_reports = {}
            team_progress = st.progress(0)
            team_status = st.empty()

            for index, agent_key in enumerate(team_order, start=1):
                agent = AGENT_DEFINITIONS[agent_key]
                team_status.info(f"Running {agent['name']} ({index}/{len(team_order)})...")

                agent_prompt = build_agent_prompt(
                    agent_key=agent_key,
                    evidence=agent_evidence,
                    user_question=agent_question,
                    rag_context=agent_rag_context,
                )

                team_reports[agent_key] = ask_qwen(
                    agent_prompt,
                    max_tokens=agent["max_tokens"],
                )
                team_progress.progress(index / (len(team_order) + 1))

            team_status.info("Running Coordinator Agent synthesis...")
            team_prompt = build_agent_team_synthesis_prompt(
                evidence=agent_evidence,
                user_question=agent_question,
                rag_context=agent_rag_context,
                agent_reports=team_reports,
            )
            team_answer = ask_qwen(team_prompt, max_tokens=9000)
            team_progress.progress(1.0)
            team_status.success("Agent Team Review completed.")

            all_team_sql_queries = []
            for report_text in list(team_reports.values()) + [team_answer]:
                for sql_query in extract_all_sql_queries(report_text):
                    if sql_query not in all_team_sql_queries:
                        all_team_sql_queries.append(sql_query)

            if (
                agent_execute_sql
                and not all_team_sql_queries
                and question_needs_sql_result(agent_question)
            ):
                with st.spinner("Agent Team answered without SQL. Asking Qwen for SQL-only so the team can finish the task..."):
                    retry_sql_query, retry_sql_answer = ask_qwen_for_sql_only(
                        user_question=agent_question,
                        selected_table=selected_table,
                        dtype_df=dtype_df,
                        rag_context=agent_rag_context,
                    )

                if retry_sql_query:
                    all_team_sql_queries = [retry_sql_query]
                    team_answer += (
                        "\n\n## SQL-Only Retry Used For Execution\n\n"
                        "The team report did not include runnable SQL, so Streamlit asked Qwen for one safe SQL draft:\n\n"
                        "```sql\n"
                        f"{retry_sql_query.strip()}\n"
                        "```"
                    )
                elif retry_sql_answer:
                    team_answer += (
                        "\n\n## SQL-Only Retry Could Not Produce Runnable SQL\n\n"
                        f"{retry_sql_answer}"
                    )

            team_sql = format_sql_drafts(all_team_sql_queries)
            team_sql_safety = build_sql_safety_summary(all_team_sql_queries)
            team_sql_execution_records = []
            team_result_answer = None

            if agent_execute_sql and all_team_sql_queries:
                with st.spinner("Running Agent Team SQL drafts through the read-only safety gate..."):
                    team_sql_execution_records = execute_agent_sql_queries(
                        sql_queries=all_team_sql_queries,
                        sql_execution_enabled=agent_can_execute_sql,
                        multi_database_sql_enabled=selection_has_common_table,
                        selected_database=selected_database,
                        selected_table_targets=selected_table_targets,
                    )

                if any(record.get("result") is not None for record in team_sql_execution_records):
                    team_result_prompt = build_agent_sql_result_prompt(
                        agent_name="Agent Team Review",
                        user_question=agent_question,
                        selected_table=selected_table,
                        selection_description=selection_description,
                        agent_answer=team_answer,
                        execution_records=team_sql_execution_records,
                    )
                    with st.spinner("Coordinator Agent is answering from actual SQL results..."):
                        team_result_answer = ask_qwen(team_result_prompt, max_tokens=6000)

            team_metadata = {
                "selection": selection_description,
                "database": selected_database,
                "table": selected_table,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "agent_question": agent_question,
                "rag_used": bool(agent_use_rag),
                "agent_sql_execution_enabled": bool(agent_execute_sql),
                "team_order": team_order,
            }

            team_report_text_to_save = team_answer
            if team_result_answer:
                team_report_text_to_save += (
                    "\n\n---\n\n"
                    "## Final Answer From Executed SQL\n\n"
                    f"{team_result_answer}"
                )

            team_file_paths = save_agent_team_report(
                team_report_text=team_report_text_to_save,
                individual_reports=team_reports,
                metadata=team_metadata,
                sql_text=team_sql,
            )

            if not MEMORY_IMPORT_ERROR:
                team_memory_paths = save_agent_task_memory({
                    "agent_key": "agent_team",
                    "agent_name": "Agent Team Review",
                    "question": agent_question,
                    "selection": selection_description,
                    "database": selected_database,
                    "table": selected_table,
                    "row_count": int(len(df)),
                    "column_count": int(len(df.columns)),
                    "rag_used": bool(agent_use_rag),
                    "sql_server_memory_used": bool(agent_use_sql_memory),
                    "sql_text": team_sql,
                    "sql_execution_summaries": summarize_sql_execution_records(team_sql_execution_records),
                    "final_answer": team_result_answer,
                    "agent_report_excerpt": safe_preview_text(team_answer, 6000),
                    "report_files": team_file_paths,
                    "team_order": team_order,
                })
                team_file_paths.update({
                    "memory_task_history": team_memory_paths["task_history"],
                    "memory_latest_task": team_memory_paths["latest_task"],
                })

            st.session_state["agent_workspace_answer"] = team_answer
            st.session_state["agent_workspace_result_answer"] = team_result_answer
            st.session_state["agent_workspace_agent_key"] = "agent_team"
            st.session_state["agent_workspace_agent_name"] = "Agent Team Review"
            st.session_state["agent_workspace_sql"] = team_sql
            st.session_state["agent_workspace_sql_safety"] = team_sql_safety
            st.session_state["agent_workspace_sql_results"] = team_sql_execution_records
            st.session_state["agent_workspace_files"] = team_file_paths
            st.session_state["agent_workspace_rag_matches"] = agent_rag_matches
            st.session_state["agent_workspace_team_reports"] = team_reports
            st.session_state["agent_workspace_error"] = None

        except Exception as e:
            st.session_state["agent_workspace_error"] = str(e)
            st.error("Could not run the Agent Team Review.")
            st.exception(e)

    if st.session_state.get("agent_workspace_error"):
        st.warning(f"Last agent error: {st.session_state['agent_workspace_error']}")

    if st.session_state.get("agent_workspace_answer"):
        if st.session_state.get("agent_workspace_result_answer"):
            st.markdown("### Agent Final Answer From SQL Results")
            st.markdown(st.session_state["agent_workspace_result_answer"])

        st.markdown(f"### {st.session_state.get('agent_workspace_agent_name', 'Agent')} Report")
        st.markdown(st.session_state["agent_workspace_answer"])

        if st.session_state.get("agent_workspace_sql"):
            st.markdown("### Agent SQL Draft")
            st.code(st.session_state["agent_workspace_sql"], language="sql")
            st.caption(
                "This SQL draft was saved for review. It is not automatically executed from the Agent Workspace."
            )

        if st.session_state.get("agent_workspace_sql_safety"):
            st.markdown("### SQL Draft Safety Check")
            st.dataframe(
                pd.DataFrame(st.session_state["agent_workspace_sql_safety"]),
                use_container_width=True,
            )

        if st.session_state.get("agent_workspace_sql_results"):
            st.markdown("### Agent SQL Execution Results")
            for record in st.session_state["agent_workspace_sql_results"]:
                label = f"Query {record['query_index']} - {record['status']}"
                with st.expander(label, expanded=record.get("result") is not None):
                    st.code(record["sql"], language="sql")

                    if record.get("target_errors"):
                        st.warning("Some selected database targets returned errors.")
                        for display_name, error_text in record["target_errors"]:
                            st.write(f"- `{display_name}`: {error_text}")

                    if record.get("error"):
                        st.warning(record["error"])

                    if record.get("result") is not None:
                        result_df = record["result"]
                        st.write(f"Rows returned: **{len(result_df)}**")
                        show_dataframe_preview(
                            result_df,
                            max_rows=QUERY_RESULT_PREVIEW_ROWS,
                            context_label=f"agent query {record['query_index']} result",
                        )
                        show_auto_result_chart(result_df)

        if st.session_state.get("agent_workspace_team_reports"):
            with st.expander("Individual specialist reports"):
                for agent_key, report_text in st.session_state["agent_workspace_team_reports"].items():
                    st.markdown(f"#### {AGENT_DEFINITIONS[agent_key]['name']}")
                    st.markdown(report_text)

        if st.session_state.get("agent_workspace_files"):
            st.markdown("### Saved Agent Report Files")
            for label, file_path in st.session_state["agent_workspace_files"].items():
                if file_path:
                    st.write(f"- `{label}`: `{file_path}`")

        if st.session_state.get("agent_workspace_rag_matches"):
            with st.expander("Agent RAG context"):
                for index, match in enumerate(st.session_state["agent_workspace_rag_matches"], start=1):
                    st.write(f"**Chunk {index}: {match['source']}**")
                    st.write(f"Distance: {match['distance']:.4f}")
                    st.write(match["text"])

    with st.expander("Saved agent history"):
        if os.path.exists(AGENT_HISTORY_DIR):
            history_files = []
            for file_name in os.listdir(AGENT_HISTORY_DIR):
                file_path = os.path.join(AGENT_HISTORY_DIR, file_name)
                if os.path.isfile(file_path):
                    history_files.append({
                        "File": file_name,
                        "Path": file_path,
                        "Modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                        "SizeKB": round(os.path.getsize(file_path) / 1024, 1),
                    })

            history_files = sorted(history_files, key=lambda row: row["Modified"], reverse=True)[:30]
            if history_files:
                st.dataframe(pd.DataFrame(history_files), use_container_width=True)
            else:
                st.info("No timestamped agent history files yet. Run an agent to create one.")
        else:
            st.info("No agent history folder yet. Run an agent to create it.")


# -----------------------------
# LOCAL AI ANALYST
# -----------------------------

st.subheader("Ask Local Qwen About This Table")
if selection_is_single_table:
    st.caption("If Qwen returns a safe read-only SELECT query, the app will run it and show the result below.")
elif selection_has_common_table:
    st.caption(
        "Multiple databases are loaded for the same schema.table. If Qwen returns a safe read-only SELECT query, "
        "the app will run that same SQL inside each selected database and combine the results."
    )
else:
    st.caption(
        "Multiple table targets are loaded. Qwen can analyze the combined data, "
        "but automatic SQL execution needs one exact table or the same schema.table across databases."
    )

default_question = """
Analyze this selected SQL table.
Explain what it likely represents, suggest KPIs, data quality checks,
possible relationships with other tables, useful charts, and dashboard ideas.
"""

user_question = st.text_area(
    "Your question for local Qwen:",
    value=default_question,
    height=150
)

if st.button("Ask Local Qwen"):
    try:
        ai_context = build_ai_table_context(df, dtype_df)
        sql_execution_enabled = selection_is_single_table or selection_has_common_table
        multi_database_sql_enabled = sql_execution_enabled and not selection_is_single_table
        selection_targets_text = (
            pd.DataFrame(selected_table_targets).to_string(index=False)
            if selected_table_targets
            else "No table target metadata available."
        )
        if selection_is_single_table:
            sql_rules = f"""
- If the user asks for SQL, return one corrected SQL Server query first.
- Put SQL inside a ```sql code block.
- The SQL code block must contain only runnable SQL, no labels, no headings, no explanation text.
- Use SQL Server T-SQL only. Do not use DATE_TRUNC, LIMIT, ILIKE, or PostgreSQL casts.
- For monthly trends, use DATEFROMPARTS(YEAR(date_column), MONTH(date_column), 1).
- If the user asks to show, get, list, find, sort, rank, filter, or calculate data, write a read-only SELECT query so the app can run it.
- Use the exact selected table name: {selected_table}
- Use only columns that exist in the provided column list.
"""
        elif selection_has_common_table:
            sql_rules = f"""
- If the user asks for SQL, return one corrected SQL Server query first.
- Put SQL inside a ```sql code block.
- The SQL code block must contain only runnable SQL, no labels, no headings, no explanation text.
- Use SQL Server T-SQL only. Do not use DATE_TRUNC, LIMIT, ILIKE, or PostgreSQL casts.
- For monthly trends, use DATEFROMPARTS(YEAR(date_column), MONTH(date_column), 1).
- The app will run this same SELECT separately inside each selected database and combine the results.
- Use the exact table name: {selected_table}
- Do not prefix the table with a database name.
- Do not use SourceDatabase or SourceTable in the SQL; Python adds those columns after each database returns results.
- Use only columns that exist in the provided column list.
"""
        else:
            sql_rules = """
- This is a combined pandas dataframe loaded from multiple table targets.
- Automatic SQL execution is disabled because there is no single active SQL table target.
- If the user asks for SQL, provide a SQL template or per-database approach, but say it must be run separately per database/table target.
- Use SourceDatabase and SourceTable columns when explaining comparisons across loaded targets.
- Use only columns that exist in the provided column list.
"""
        rag_matches = []
        rag_context = "RAG knowledge base is disabled."
        rag_error = None

        if use_rag:
            try:
                with st.spinner("Searching local ChromaDB knowledge base..."):
                    rag_matches = search_knowledge(user_question, top_k=4)
                rag_context = format_retrieved_context(rag_matches)
            except Exception as rag_exception:
                rag_error = str(rag_exception)
                rag_context = f"RAG search failed: {rag_error}"
                st.warning(rag_context)

        prompt = f"""
You are a database analyst specializing in irrigation, GIS, and infrastructure management systems.
Answer directly. Do not spend many tokens on internal reasoning. Keep the final answer practical and concise.

Context:
This is a SQL Server database explored through a local Streamlit app.
The selected table may belong to an irrigation or Water Users Association management system.

Database:
{selected_database}

Table:
{selected_table}

Selection:
{selection_description}

Selected table targets:
{selection_targets_text}

Automatic SQL execution enabled:
{sql_execution_enabled}

Multi-database SQL execution enabled:
{multi_database_sql_enabled}

Rows loaded in Streamlit:
{len(df)}

Total columns in loaded table:
{len(df.columns)}

Column names, capped for local model context:
{ai_context["all_columns_text"]}

Columns included in the detailed AI sample:
{ai_context["included_columns"]}

Omitted columns from detailed sample:
{ai_context["omitted_columns"]}

Column data types and missing values for included columns:
{ai_context["dtypes_text"]}

Missing values summary:
{ai_context["missing_text"]}

Sample rows:
{ai_context["sample_rows"]}

Retrieved RAG knowledge context:
{rag_context}

User question:
{user_question}

Main instruction:
Answer the user's question first and directly.

Rules:
{sql_rules}
- Do not invent table names or columns.
- Do not use English words from the user question as column names unless they appear in the provided column list.
- If a requested column does not exist, derive it from an existing column only when obvious, otherwise say it does not exist.
- Add TOP 100 or a smaller TOP value unless the user asks for a different number.
- For SQL questions, add only a short explanation after the query, maximum 3 bullets.
- Do not add KPIs, dashboard ideas, relationships, machine learning, or broad table analysis unless the user specifically asks for those things.
- If the user asks for broad analysis, then provide the broader analysis.
- Be practical and beginner-friendly.
"""

        context_note = (
            f"Sending AI context: {ai_context['sample_row_count']} sample rows, "
            f"{len(ai_context['included_columns'])} detailed columns, "
            f"{ai_context['omitted_columns']} columns omitted from the detailed sample, "
            f"{ai_context['max_cell_length']} max characters per sample cell, "
            f"{ai_context['max_tokens']} max response tokens."
        )
        st.caption(context_note)

        with st.spinner("Checking LM Studio and asking local Qwen... Long local GPU answers can take several minutes."):
            check_lmstudio_ready()
            answer = ask_qwen(prompt, max_tokens=ai_context["max_tokens"])

        sql_query = extract_sql_query(answer)
        sql_result = None
        sql_error = None
        original_sql_query = sql_query
        repaired_sql = False
        fallback_sql_used = False
        repair_answer = None
        sql_only_answer = None
        sql_target_errors = []

        def run_sql_for_current_selection(current_sql_query):
            if multi_database_sql_enabled:
                return run_read_only_query_for_table_targets(
                    selected_table_targets,
                    current_sql_query,
                )

            return run_read_only_query(selected_database, current_sql_query), []

        if (
            sql_execution_enabled
            and not sql_query
            and question_needs_sql_result(user_question)
        ):
            try:
                with st.spinner("Qwen answered without SQL. Asking again for SQL only..."):
                    sql_query, sql_only_answer = ask_qwen_for_sql_only(
                        user_question=user_question,
                        selected_table=selected_table,
                        dtype_df=dtype_df,
                        rag_context=rag_context,
                    )
                    original_sql_query = sql_query
            except Exception as sql_only_exception:
                sql_error = f"Qwen did not provide SQL, and the SQL-only retry failed: {sql_only_exception}"

        if sql_query and not sql_execution_enabled:
            sql_error = (
                "The SQL query was not run because multiple table targets are loaded. "
                "Load one exact table if you want Streamlit to run Qwen SQL automatically."
            )
        elif sql_query:
            sql_query = normalize_sql_server_dialect(sql_query)
            try:
                spinner_text = (
                    "Running Qwen's safe read-only SQL query in each selected database..."
                    if multi_database_sql_enabled
                    else "Running Qwen's safe read-only SQL query..."
                )
                with st.spinner(spinner_text):
                    sql_result, sql_target_errors = run_sql_for_current_selection(sql_query)
            except Exception as sql_exception:
                sql_error = str(sql_exception)

                try:
                    with st.spinner("SQL Server rejected the query. Asking Qwen to repair it and retry..."):
                        repaired_query, repair_answer = repair_sql_query(
                            original_sql=sql_query,
                            sql_error=sql_error,
                            selected_table=selected_table,
                            dtype_df=dtype_df,
                            user_question=user_question,
                        )

                    if repaired_query and repaired_query != sql_query:
                        sql_query = normalize_sql_server_dialect(repaired_query)
                        sql_result, sql_target_errors = run_sql_for_current_selection(sql_query)
                        sql_error = None
                        repaired_sql = True
                    elif repaired_query:
                        sql_error = f"Qwen returned the same SQL again. Original error: {sql_error}"
                    else:
                        sql_error = f"Qwen could not produce a repaired SQL query. Original error: {sql_error}"
                except Exception as repair_exception:
                    sql_error = (
                        f"Original SQL failed: {sql_error}\n\n"
                        f"Automatic repair also failed: {repair_exception}"
                    )

                if sql_error:
                    fallback_query = build_revenue_trend_sql(user_question, selected_table, dtype_df)
                    if fallback_query:
                        try:
                            with st.spinner("Using SQL Server revenue trend fallback query..."):
                                sql_query = fallback_query
                                sql_result, sql_target_errors = run_sql_for_current_selection(sql_query)
                            sql_error = None
                            fallback_sql_used = True
                        except Exception as fallback_exception:
                            sql_error = (
                                f"{sql_error}\n\n"
                                f"Revenue trend fallback also failed: {fallback_exception}"
                            )
        elif sql_execution_enabled:
            fallback_query = build_revenue_trend_sql(user_question, selected_table, dtype_df)
            if fallback_query:
                try:
                    with st.spinner("Using SQL Server revenue trend fallback query..."):
                        sql_query = fallback_query
                        sql_result, sql_target_errors = run_sql_for_current_selection(sql_query)
                    sql_error = None
                    fallback_sql_used = True
                except Exception as fallback_exception:
                    sql_error = f"Revenue trend fallback failed: {fallback_exception}"

        st.session_state["qwen_answer"] = answer
        st.session_state["qwen_question"] = user_question
        st.session_state["qwen_context_note"] = context_note
        st.session_state["qwen_error"] = None
        st.session_state["qwen_sql"] = sql_query
        st.session_state["qwen_original_sql"] = original_sql_query
        st.session_state["qwen_sql_repaired"] = repaired_sql
        st.session_state["qwen_sql_fallback_used"] = fallback_sql_used
        st.session_state["qwen_sql_repair_answer"] = repair_answer
        st.session_state["qwen_sql_only_answer"] = sql_only_answer
        st.session_state["qwen_sql_result"] = sql_result
        st.session_state["qwen_sql_error"] = sql_error
        st.session_state["qwen_sql_target_errors"] = sql_target_errors
        st.session_state["rag_matches"] = rag_matches
        st.session_state["rag_context"] = rag_context
        st.session_state["rag_error"] = rag_error
        st.session_state["rag_enabled"] = use_rag

    except Exception as e:
        st.session_state["qwen_error"] = str(e)
        st.error("Could not get response from local Qwen.")
        st.exception(e)

if st.session_state.get("rag_enabled"):
    st.subheader("Retrieved Knowledge Context")

    if st.session_state.get("rag_error"):
        st.warning(st.session_state["rag_error"])
    elif st.session_state.get("rag_matches"):
        for index, match in enumerate(st.session_state["rag_matches"], start=1):
            with st.expander(f"Chunk {index}: {match['source']}"):
                st.write(f"Distance: {match['distance']:.4f}")
                st.write(match["text"])
    else:
        st.info("No relevant ChromaDB knowledge chunks were retrieved.")

if st.session_state.get("qwen_answer"):
    st.markdown("### Qwen Analysis")
    st.markdown(st.session_state["qwen_answer"])

    if st.session_state.get("qwen_sql"):
        st.markdown("### Qwen SQL Result")

        if st.session_state.get("qwen_sql_only_answer"):
            st.info("Qwen's first answer did not include runnable SQL, so the app asked again for SQL only.")

        if st.session_state.get("qwen_sql_repaired"):
            st.info("The first SQL query failed, so Qwen repaired it automatically before running.")
            with st.expander("Original failed SQL"):
                st.code(st.session_state.get("qwen_original_sql", ""), language="sql")

        if st.session_state.get("qwen_sql_fallback_used"):
            st.info("The app used a SQL Server-safe revenue trend fallback query for this chart request.")

        st.code(st.session_state["qwen_sql"], language="sql")

        if st.session_state.get("qwen_sql_error"):
            st.warning(f"The SQL query was not run: {st.session_state['qwen_sql_error']}")

        if st.session_state.get("qwen_sql_target_errors"):
            with st.expander("Database targets with SQL errors"):
                for display_name, error_text in st.session_state["qwen_sql_target_errors"]:
                    st.warning(f"{display_name}: {error_text}")

        if st.session_state.get("qwen_sql_result") is not None:
            result_df = st.session_state["qwen_sql_result"]
            st.write(f"Rows returned: **{len(result_df)}**")
            show_dataframe_preview(
                result_df,
                max_rows=QUERY_RESULT_PREVIEW_ROWS,
                context_label="query result",
            )
            show_auto_result_chart(result_df)
        elif not st.session_state.get("qwen_sql_error"):
            st.info("Qwen wrote SQL, but there was no result to display.")
    else:
        st.info("No runnable SQL query was found in Qwen's answer.")

    with st.expander("Last Qwen request details"):
        st.write(f"**Question:** {st.session_state.get('qwen_question', '')}")
        st.write(st.session_state.get("qwen_context_note", ""))

if st.session_state.get("qwen_error"):
    st.warning(f"Last Qwen error: {st.session_state['qwen_error']}")
