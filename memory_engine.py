import json
import os
import re
from datetime import datetime


MEMORY_DIR = os.path.join("outputs", "agent_memory")
TASK_HISTORY_PATH = os.path.join(MEMORY_DIR, "agent_task_history.jsonl")
DATABASE_CONTEXT_JSON_PATH = os.path.join(MEMORY_DIR, "sql_server_context.json")
DATABASE_CONTEXT_MD_PATH = os.path.join(MEMORY_DIR, "sql_server_context.md")


CONCEPT_KEYWORDS = {
    "payments": ["payment", "payments", "pay", "paid", "payer", "money", "amount", "fee", "cash", "collection", "repayment", "revenue"],
    "water_delivery": ["water", "delivery", "delivered", "volume", "hectare", "irrigation", "usage", "service", "exposure"],
    "users_customers": ["user", "users", "customer", "customers", "mashak", "farmer", "farmers", "client", "clients", "borrower", "borrowers", "person", "people"],
    "contracts": ["contract", "contracts", "kod", "code", "agreement", "identifier", "identification", "account", "reference"],
    "wua_branch_segment": ["wua", "branch", "region", "community", "system", "segment", "portfolio", "group", "unit"],
    "dates_time": ["date", "time", "year", "month", "period"],
    "contact_quality": ["phone", "address", "passport", "email", "contact", "mobile", "telephone"],
    "debt_risk": ["debt", "overdue", "loan", "balance", "arrear", "arrears", "unpaid", "delinquent", "risk"],
    "identity_keys": ["id", "kod", "code", "number", "guid", "identifier", "key", "unique", "distinct"],
}


QUERY_SYNONYMS = {
    "paying": ["payment", "payments", "pay", "paid", "payer", "collection", "repayment", "revenue"],
    "payer": ["payment", "payments", "pay", "paid", "customer", "user", "contract"],
    "revenue": ["payment", "money", "amount", "collection", "repayment"],
    "collection": ["payment", "pay", "money", "amount", "repayment", "revenue"],
    "repayment": ["payment", "pay", "money", "amount", "collection"],
    "wateruser": ["wateruser", "water", "user", "customer", "farmer", "mashak"],
    "waterusers": ["wateruser", "water", "user", "customer", "farmer", "mashak"],
    "customer": ["user", "wateruser", "farmer", "mashak", "client", "borrower"],
    "customers": ["user", "users", "wateruser", "farmer", "mashak", "client", "borrower"],
    "borrower": ["customer", "user", "wateruser", "farmer", "mashak"],
    "identifier": ["id", "kod", "code", "key", "number", "contract"],
    "identification": ["id", "kod", "code", "key", "number", "contract"],
    "unique": ["distinct", "id", "kod", "code", "identifier", "count"],
    "distinct": ["unique", "id", "kod", "code", "identifier", "count"],
    "month": ["date", "time", "period", "year"],
    "monthly": ["month", "date", "time", "period", "year"],
    "portfolio": ["wua", "branch", "segment", "group", "region", "system"],
    "branch": ["wua", "segment", "group", "region", "system"],
    "risk": ["debt", "overdue", "unpaid", "missing", "quality", "anomaly"],
    "dashboard": ["powerbi", "visual", "chart", "kpi", "report"],
    "chart": ["dashboard", "visual", "powerbi", "trend", "kpi"],
}


def ensure_memory_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return MEMORY_DIR


def safe_json_text(value, max_length=12000):
    text = "" if value is None else str(value)
    if len(text) > max_length:
        return text[:max_length] + "\n...[truncated]"
    return text


def summarize_sql_execution_records(execution_records):
    summaries = []

    for record in execution_records or []:
        result_df = record.get("result")
        summary = {
            "query_index": record.get("query_index"),
            "read_only_safe": bool(record.get("read_only_safe")),
            "status": safe_json_text(record.get("status"), 1000),
            "sql": safe_json_text(record.get("sql"), 8000),
            "error": safe_json_text(record.get("error"), 2000),
            "target_error_count": len(record.get("target_errors") or []),
            "result_row_count": None,
            "result_columns": [],
        }

        if result_df is not None:
            summary["result_row_count"] = int(len(result_df))
            summary["result_columns"] = [str(column) for column in result_df.columns]

        summaries.append(summary)

    return summaries


def save_agent_task_memory(record):
    ensure_memory_dir()

    stored_record = dict(record)
    stored_record.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))

    with open(TASK_HISTORY_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(stored_record, ensure_ascii=False) + "\n")

    latest_path = os.path.join(MEMORY_DIR, "latest_agent_task.json")
    with open(latest_path, "w", encoding="utf-8") as file:
        json.dump(stored_record, file, ensure_ascii=False, indent=2)

    return {
        "task_history": TASK_HISTORY_PATH,
        "latest_task": latest_path,
    }


def load_agent_task_history(search_text=None, limit=50):
    if not os.path.exists(TASK_HISTORY_PATH):
        return []

    search_text = (search_text or "").strip().lower()
    records = []

    with open(TASK_HISTORY_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if search_text:
                searchable = json.dumps(record, ensure_ascii=False).lower()
                if search_text not in searchable:
                    continue

            records.append(record)

    records = list(reversed(records))
    return records[:limit]


def detect_concepts(table_name, columns):
    haystack = " ".join(
        [str(table_name)] + [str(column.get("name", "")) for column in columns]
    ).lower()

    concepts = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            concepts.append(concept)

    return concepts


def detect_key_columns(columns):
    key_columns = []

    for column in columns:
        column_name = str(column.get("name", ""))
        lowered = column_name.lower()
        if (
            lowered.endswith("id")
            or lowered.endswith("kod")
            or lowered.endswith("code")
            or "id" == lowered
            or "kod" in lowered
            or "contract" in lowered
        ):
            key_columns.append(column_name)

    return key_columns


def scan_sql_server_context(database_names, connect_to_database):
    ensure_memory_dir()

    context = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_type": "metadata_only_read_only",
        "privacy_note": (
            "This memory stores SQL Server metadata, row-count estimates, table names, "
            "columns, data types, and detected concepts. It does not store raw table rows."
        ),
        "databases": [],
        "errors": [],
        "totals": {
            "database_count": 0,
            "table_count": 0,
            "column_count": 0,
            "estimated_rows": 0,
        },
        "concept_index": {},
    }

    for database_name in database_names:
        conn = None
        try:
            conn = connect_to_database(database_name)

            tables_df = _read_sql(
                conn,
                """
                SELECT
                    s.name AS table_schema,
                    t.name AS table_name,
                    SUM(CASE WHEN p.index_id IN (0, 1) THEN p.rows ELSE 0 END) AS estimated_rows
                FROM sys.tables t
                INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                LEFT JOIN sys.partitions p ON t.object_id = p.object_id
                GROUP BY s.name, t.name
                ORDER BY s.name, t.name
                """,
            )

            columns_df = _read_sql(
                conn,
                """
                SELECT
                    TABLE_SCHEMA AS table_schema,
                    TABLE_NAME AS table_name,
                    COLUMN_NAME AS column_name,
                    DATA_TYPE AS data_type,
                    IS_NULLABLE AS is_nullable,
                    CHARACTER_MAXIMUM_LENGTH AS character_maximum_length,
                    NUMERIC_PRECISION AS numeric_precision,
                    NUMERIC_SCALE AS numeric_scale,
                    ORDINAL_POSITION AS ordinal_position
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
                """,
            )

            database_context = {
                "database": database_name,
                "table_count": int(len(tables_df)),
                "column_count": int(len(columns_df)),
                "estimated_rows": int(tables_df["estimated_rows"].fillna(0).sum()) if len(tables_df) else 0,
                "tables": [],
            }

            for _, table_row in tables_df.iterrows():
                schema_name = str(table_row["table_schema"])
                table_name = str(table_row["table_name"])
                full_table_name = f"{schema_name}.{table_name}"
                table_columns_df = columns_df[
                    (columns_df["table_schema"] == schema_name)
                    & (columns_df["table_name"] == table_name)
                ]

                columns = []
                for _, column_row in table_columns_df.iterrows():
                    columns.append({
                        "name": str(column_row["column_name"]),
                        "data_type": str(column_row["data_type"]),
                        "nullable": str(column_row["is_nullable"]),
                        "ordinal": int(column_row["ordinal_position"]),
                    })

                concepts = detect_concepts(full_table_name, columns)
                key_columns = detect_key_columns(columns)
                estimated_rows = int(table_row["estimated_rows"] or 0)

                table_context = {
                    "schema": schema_name,
                    "table": table_name,
                    "full_name": full_table_name,
                    "estimated_rows": estimated_rows,
                    "column_count": len(columns),
                    "columns": columns,
                    "key_like_columns": key_columns,
                    "detected_concepts": concepts,
                }

                database_context["tables"].append(table_context)

                for concept in concepts:
                    context["concept_index"].setdefault(concept, []).append({
                        "database": database_name,
                        "table": full_table_name,
                        "estimated_rows": estimated_rows,
                        "key_like_columns": key_columns[:10],
                    })

            context["databases"].append(database_context)

        except Exception as exc:
            context["errors"].append({
                "database": database_name,
                "error": str(exc),
            })
        finally:
            if conn is not None:
                conn.close()

    context["totals"]["database_count"] = len(context["databases"])
    context["totals"]["table_count"] = sum(db["table_count"] for db in context["databases"])
    context["totals"]["column_count"] = sum(db["column_count"] for db in context["databases"])
    context["totals"]["estimated_rows"] = sum(db["estimated_rows"] for db in context["databases"])

    save_database_context(context)
    return context


def _read_sql(conn, query):
    # Keep pandas imported locally so importing memory_engine stays lightweight.
    import pandas as pd

    return pd.read_sql(query, conn)


def save_database_context(context):
    ensure_memory_dir()

    with open(DATABASE_CONTEXT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)

    with open(DATABASE_CONTEXT_MD_PATH, "w", encoding="utf-8") as file:
        file.write(build_database_context_markdown(context))

    return {
        "json": DATABASE_CONTEXT_JSON_PATH,
        "markdown": DATABASE_CONTEXT_MD_PATH,
    }


def load_database_context():
    if not os.path.exists(DATABASE_CONTEXT_JSON_PATH):
        return None

    with open(DATABASE_CONTEXT_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_database_context_markdown(context):
    lines = [
        "# SQL Server Context Memory",
        "",
        f"Generated at: {context.get('generated_at', '')}",
        "",
        context.get("privacy_note", ""),
        "",
        "## Totals",
        "",
        f"- Databases scanned: {context['totals']['database_count']}",
        f"- Tables scanned: {context['totals']['table_count']}",
        f"- Columns scanned: {context['totals']['column_count']}",
        f"- Estimated rows: {context['totals']['estimated_rows']:,}",
        "",
        "## Databases",
        "",
    ]

    for database in context.get("databases", []):
        lines.extend([
            f"### {database['database']}",
            "",
            f"- Tables: {database['table_count']}",
            f"- Columns: {database['column_count']}",
            f"- Estimated rows: {database['estimated_rows']:,}",
            "",
        ])

        top_tables = sorted(
            database.get("tables", []),
            key=lambda table: table.get("estimated_rows", 0),
            reverse=True,
        )[:15]

        for table in top_tables:
            concepts = ", ".join(table.get("detected_concepts", [])) or "none"
            keys = ", ".join(table.get("key_like_columns", [])[:8]) or "none"
            lines.append(
                f"- `{table['full_name']}`: rows ~{table['estimated_rows']:,}, "
                f"columns {table['column_count']}, concepts: {concepts}, key-like columns: {keys}"
            )

        lines.append("")

    if context.get("errors"):
        lines.extend(["## Scan Errors", ""])
        for error in context["errors"]:
            lines.append(f"- `{error['database']}`: {error['error']}")
        lines.append("")

    return "\n".join(lines)


def tokenize_text(value):
    separators = [",", ".", ";", ":", "/", "\\", "-", "_", "(", ")", "[", "]", "{", "}"]
    text = str(value or "").lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value or ""))
    text = text.lower()
    for separator in separators:
        text = text.replace(separator, " ")
    return [token for token in text.split() if token]


def expand_query_terms(query):
    base_terms = tokenize_text(query)

    if not base_terms:
        base_terms = ["payment", "water", "wua", "contract", "date", "money"]

    expanded_terms = set(base_terms)
    expanded_concepts = set()

    joined_query = " ".join(base_terms)

    for term in list(base_terms):
        for synonym in QUERY_SYNONYMS.get(term, []):
            expanded_terms.add(synonym.lower())

        for concept, keywords in CONCEPT_KEYWORDS.items():
            if term == concept or term in keywords:
                expanded_concepts.add(concept)
                for keyword in keywords:
                    expanded_terms.add(keyword.lower())

    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(keyword in joined_query for keyword in keywords):
            expanded_concepts.add(concept)
            for keyword in keywords:
                expanded_terms.add(keyword.lower())

    return sorted(expanded_terms), sorted(expanded_concepts)


def term_matches_token(term, token):
    if term == token:
        return True

    normalized_term = term[:-1] if term.endswith("s") and len(term) > 4 else term
    normalized_token = token[:-1] if token.endswith("s") and len(token) > 4 else token

    if normalized_term == normalized_token:
        return True

    if (
        len(normalized_term) >= 4
        and len(normalized_token) >= 4
        and (
            normalized_token.startswith(normalized_term)
            or normalized_term.startswith(normalized_token)
        )
    ):
        return True

    return False


def score_context_match(query_terms, query_concepts, database_name, table, original_query):
    original_query = str(original_query or "").lower()
    database_name_lower = str(database_name or "").lower()
    table_name = str(table.get("full_name", "")).lower()
    table_base_name = table_name.split(".")[-1]
    columns = [str(column["name"]).lower() for column in table.get("columns", [])]
    key_columns = [str(column).lower() for column in table.get("key_like_columns", [])]
    concepts = [str(concept).lower() for concept in table.get("detected_concepts", [])]

    database_tokens = tokenize_text(database_name_lower)
    table_tokens = tokenize_text(table_name)
    table_base_tokens = tokenize_text(table_base_name)
    column_tokens = []
    for column in columns:
        column_tokens.extend(tokenize_text(column))

    score = 0
    matched_terms = []
    matched_concepts = []

    # Exact names matter when the user explicitly names a table or database.
    # This prevents broad tables such as WaterUserBlock from hiding exact dbo.Water.
    if table_name in original_query:
        score += 60
        matched_terms.append(table_name)

    if f"dbo.{table_base_name}" in original_query:
        score += 50
        matched_terms.append(f"dbo.{table_base_name}")

    if table_base_name in query_terms:
        score += 35
        matched_terms.append(table_base_name)

    if database_name_lower in original_query:
        score += 25
        matched_terms.append(database_name_lower)

    for term in query_terms:
        term_score = 0

        if any(term_matches_token(term, token) for token in database_tokens):
            term_score += 8

        if term == table_base_name:
            term_score += 35

        if any(term_matches_token(term, token) for token in table_tokens):
            term_score += 6

        if any(term_matches_token(term, token) for token in table_base_tokens):
            term_score += 5

        if any(term_matches_token(term, token) for token in column_tokens):
            term_score += 3

        key_column_tokens = []
        for key_column in key_columns:
            key_column_tokens.extend(tokenize_text(key_column))

        if any(term_matches_token(term, token) for token in key_column_tokens):
            term_score += 2

        if any(term in concept for concept in concepts):
            term_score += 2

        if term_score:
            matched_terms.append(term)
            score += term_score

    for concept in query_concepts:
        if concept in concepts:
            matched_concepts.append(concept)
            score += 8

    if matched_terms and table.get("estimated_rows", 0) > 0:
        score += 1

    return score, matched_terms, matched_concepts


def search_database_context(query, max_results=30):
    context = load_database_context()
    if not context:
        return []

    query_terms, query_concepts = expand_query_terms(query)
    matches = []

    for database in context.get("databases", []):
        for table in database.get("tables", []):
            score, matched_terms, matched_concepts = score_context_match(
                query_terms=query_terms,
                query_concepts=query_concepts,
                database_name=database["database"],
                table=table,
                original_query=query,
            )

            if score <= 0:
                continue

            matches.append({
                "score": score,
                "database": database["database"],
                "table": table["full_name"],
                "estimated_rows": table.get("estimated_rows", 0),
                "column_count": table.get("column_count", 0),
                "detected_concepts": table.get("detected_concepts", []),
                "key_like_columns": table.get("key_like_columns", []),
                "columns": [column["name"] for column in table.get("columns", [])],
                "matched_terms": matched_terms[:20],
                "matched_concepts": matched_concepts,
            })

    matches.sort(key=lambda row: (row["score"], row["estimated_rows"]), reverse=True)
    return matches[:max_results]


def format_database_context_for_prompt(query, max_results=12, max_chars=8000):
    matches = search_database_context(query, max_results=max_results)
    if not matches:
        return "No SQL Server context memory found for this question."

    lines = [
        "SQL Server context memory matches. This is metadata only, not raw data rows.",
        "",
    ]

    for match in matches:
        lines.extend([
            f"Database: {match['database']}",
            f"Table: {match['table']}",
            f"Estimated rows: {match['estimated_rows']:,}",
            f"Detected concepts: {', '.join(match['detected_concepts']) or 'none'}",
            f"Why matched: terms={', '.join(match.get('matched_terms', [])[:12]) or 'none'}; concepts={', '.join(match.get('matched_concepts', [])) or 'none'}",
            f"Key-like columns: {', '.join(match['key_like_columns'][:10]) or 'none'}",
            f"Columns: {', '.join(match['columns'][:40])}",
            "",
        ])

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def get_memory_stats():
    context = load_database_context()
    task_count = 0

    if os.path.exists(TASK_HISTORY_PATH):
        with open(TASK_HISTORY_PATH, "r", encoding="utf-8") as file:
            task_count = sum(1 for line in file if line.strip())

    stats = {
        "task_count": task_count,
        "has_database_context": bool(context),
        "database_count": 0,
        "table_count": 0,
        "column_count": 0,
        "estimated_rows": 0,
        "context_generated_at": None,
    }

    if context:
        stats.update({
            "database_count": context["totals"]["database_count"],
            "table_count": context["totals"]["table_count"],
            "column_count": context["totals"]["column_count"],
            "estimated_rows": context["totals"]["estimated_rows"],
            "context_generated_at": context.get("generated_at"),
        })

    return stats
