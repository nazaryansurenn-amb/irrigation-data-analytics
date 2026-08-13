import json
import os
from datetime import datetime


AGENT_REPORTS_DIR = os.path.join("outputs", "agent_reports")
AGENT_HISTORY_DIR = os.path.join(AGENT_REPORTS_DIR, "history")


AGENT_TEAM_ORDER = [
    "senior_analytics_lead",
    "data_profiler",
    "data_modeler",
    "sql_analyst",
    "anomaly_detection",
    "risk_mis",
    "powerbi",
    "validator",
]


AGENT_DEFINITIONS = {
    "senior_analytics_lead": {
        "name": "Senior Analytics Lead Agent",
        "button": "Run Senior Analytics Lead",
        "max_tokens": 7000,
        "report_file": "senior_analytics_lead_report.md",
        "role": (
            "You are a Senior Analytics Lead Agent. Your job is to turn messy business "
            "questions into a rigorous analytics plan, identify the right metrics, "
            "segments, SQL checks, validation risks, and executive interpretation."
        ),
        "focus": [
            "business question framing",
            "hypothesis-driven analysis",
            "KPI design and metric definitions",
            "segmentation and cohort thinking",
            "what SQL must be run and why",
            "executive-ready interpretation",
        ],
    },
    "data_profiler": {
        "name": "Data Profiler Agent",
        "button": "Run Data Profiler Agent",
        "max_tokens": 5000,
        "report_file": "data_profiler_report.md",
        "role": (
            "You are a Data Profiler Agent. Your job is to explain the loaded SQL table "
            "from a pandas/data-quality perspective."
        ),
        "focus": [
            "table meaning from columns and sample rows",
            "column types and important fields",
            "missing values and duplicates",
            "data quality risks",
            "beginner-friendly pandas learning notes",
        ],
    },
    "sql_analyst": {
        "name": "SQL Analyst Agent",
        "button": "Run SQL Analyst Agent",
        "max_tokens": 5000,
        "report_file": "sql_analyst_report.md",
        "role": (
            "You are a SQL Server Analyst Agent. Your job is to write safe, read-only "
            "SQL Server queries and explain what they teach."
        ),
        "focus": [
            "SQL Server SELECT queries only",
            "GROUP BY, ORDER BY, TOP, aggregation, and date grouping",
            "chart-ready result sets",
            "short query explanation",
            "beginner-friendly SQL learning notes",
        ],
    },
    "data_modeler": {
        "name": "Data Model / Join Discovery Agent",
        "button": "Run Data Model Agent",
        "max_tokens": 6500,
        "report_file": "data_model_join_discovery_report.md",
        "role": (
            "You are a Data Model and Join Discovery Agent. Your job is to infer likely "
            "table relationships from SQL Server metadata, key-like columns, business "
            "concepts, and sample evidence, then propose safe join paths to verify."
        ),
        "focus": [
            "likely primary keys and foreign-key-like columns",
            "join path hypotheses",
            "grain of each table",
            "one-to-many and many-to-one relationship risks",
            "SQL checks to validate joins",
            "star schema / Power BI model ideas",
        ],
    },
    "anomaly_detection": {
        "name": "Anomaly Detection Agent",
        "button": "Run Anomaly Detection Agent",
        "max_tokens": 6500,
        "report_file": "anomaly_detection_report.md",
        "role": (
            "You are an Anomaly Detection Agent. Your job is to find suspicious patterns, "
            "outliers, sudden changes, mismatches, duplicate behavior, and data quality "
            "signals using safe SQL and practical analytics reasoning."
        ),
        "focus": [
            "outliers and unusual amounts",
            "duplicate or repeated behavior",
            "trend breaks and seasonality shifts",
            "payment versus usage mismatch",
            "missing or invalid identifiers",
            "SQL checks for anomaly candidates",
        ],
    },
    "risk_mis": {
        "name": "Operations Analytics Agent",
        "button": "Run Operations Analytics Agent",
        "max_tokens": 6000,
        "report_file": "operations_analytics_report.md",
        "role": (
            "You are an Operations Analytics Agent. Your job is to interpret "
            "irrigation/WUA data using BI, KPI monitoring, data quality, service "
            "usage, payment collection, and anomaly thinking."
        ),
        "focus": [
            "service users and customer-like entities",
            "WUA as organization unit, segment, or service group",
            "payments as collection and revenue signals",
            "water delivery as service usage",
            "data quality and contact completeness",
            "segment concentration and anomaly thinking",
        ],
    },
    "powerbi": {
        "name": "Power BI Agent",
        "button": "Run Power BI Agent",
        "max_tokens": 6000,
        "report_file": "powerbi_agent_recommendations.md",
        "role": (
            "You are a Power BI Agent. Your job is to turn the current data evidence "
            "into a dashboard plan, DAX ideas, Power Query guidance, and report story."
        ),
        "focus": [
            "Power BI pages",
            "visuals and slicers",
            "DAX measures",
            "Power Query preparation",
            "dashboard storytelling",
            "Power BI training notes",
        ],
    },
    "rag_knowledge": {
        "name": "RAG Knowledge Agent",
        "button": "Run RAG Knowledge Agent",
        "max_tokens": 5000,
        "report_file": "rag_knowledge_report.md",
        "role": (
            "You are a RAG Knowledge Agent. Your job is to use retrieved local knowledge "
            "chunks as evidence and connect them to the selected table."
        ),
        "focus": [
            "retrieved local document context",
            "evidence-based observations",
            "source-grounded training notes",
            "what the local documents add to the analysis",
        ],
    },
    "validator": {
        "name": "Validator Agent",
        "button": "Run Validator Agent",
        "max_tokens": 5000,
        "report_file": "validator_report.md",
        "role": (
            "You are a Validator Agent. Your job is to check whether the analysis is safe, "
            "realistic, and grounded in the provided columns and evidence."
        ),
        "focus": [
            "invented column warnings",
            "SQL safety",
            "missing assumptions",
            "business logic realism",
            "what should be verified before trusting the output",
        ],
    },
    "full_review": {
        "name": "Full Agent Review",
        "button": "Run Full Agent Review",
        "max_tokens": 9000,
        "report_file": "agent_full_report.md",
        "role": (
            "You are a Coordinator Agent combining the Data Profiler, SQL Analyst, "
            "Operations Analytics, Power BI, RAG Knowledge, and Validator perspectives into one report."
        ),
        "focus": [
            "integrated executive summary",
            "data profile",
            "safe SQL ideas",
            "operational analytics interpretation",
            "Power BI plan",
            "validator checklist",
            "training notes",
        ],
    },
}


def build_agent_prompt(agent_key, evidence, user_question, rag_context):
    agent = AGENT_DEFINITIONS[agent_key]
    focus_text = "\n".join(f"- {item}" for item in agent["focus"])

    return f"""
{agent["role"]}

You are working inside a local/offline Streamlit analytics project.
All SQL Server access must remain read-only. Do not suggest INSERT, UPDATE,
DELETE, DROP, ALTER, CREATE, EXEC, or destructive commands.

Agent focus:
{focus_text}

Senior data analytics standard:
- Identify the business question and the grain of the data.
- Separate facts visible in evidence from assumptions that need verification.
- Define clear metrics before interpreting them.
- Use segmentation where useful: time, WUA/unit/segment, service user/contract,
  table source, payment type, geography, or operational group.
- Prefer SQL that produces compact analytic result sets, not raw dumps.
- Recommend validation checks before trusting any insight.
- Explain both technical meaning and business meaning.
- When relevant, suggest how the result would appear in Power BI.

User question:
{user_question}

Current data evidence:

Selection:
{evidence["selection_description"]}

Database:
{evidence["selected_database"]}

Table:
{evidence["selected_table"]}

Loaded rows:
{evidence["row_count"]}

Column count:
{evidence["column_count"]}

Selected table targets:
{evidence["selected_targets_text"]}

Columns:
{evidence["columns_text"]}

Column data types and missing values:
{evidence["dtypes_text"]}

Missing values summary:
{evidence["missing_text"]}

Sample rows:
{evidence["sample_rows"]}

Operational BI summary, if available:
{evidence["risk_summary_text"]}

Power BI output files, if available:
{evidence["powerbi_files_text"]}

Agent SQL execution rules:
{evidence["sql_rules_text"]}

SQL Server context memory:
{evidence.get("sql_server_memory_text", "SQL Server context memory was not provided.")}

Retrieved RAG context:
{rag_context}

Required answer format:

## Agent Answer
Give the practical answer for this agent's role.

## Analytics Frame
State the data grain, likely business question, main metric definitions, and useful segments.

## Agent Scorecard
Give 3 to 6 scored observations using this format:
- Item: score/10 - short reason

## Recommended Actions
List the next checks, queries, visuals, or exports the user should do.

## SQL Drafts
If SQL is useful for this agent, provide one or more SQL Server SELECT queries.
If SQL is not useful, write: No SQL draft needed for this agent.
If the user's task asks for a count, total, ranking, comparison, trend, or filtered
list, SQL is useful and you should provide a runnable SQL draft.

## What You Are Learning
Explain the skill being trained in beginner-friendly terms.

## Safety / Validation Notes
Mention missing assumptions, invented-column risks, and read-only SQL safety.

## Executive Interpretation
Explain what a non-technical manager should take from this.

## Project Artifact Ideas
List files, charts, reports, or Power BI pages this agent recommends saving.

If you write SQL:
- Use SQL Server T-SQL only.
- Put SQL inside a ```sql code block.
- Use only columns listed in the evidence.
- Keep it read-only.
- Do not search only exact words from the user's command. Think wider using
  synonyms, business concepts, key-like columns, and SQL Server context memory.
  For example, "paying users" can involve payments, contracts, user/customer
  identifiers, collection, service usage, and WUA/segment grouping.
""".strip()


def build_agent_team_synthesis_prompt(evidence, user_question, rag_context, agent_reports):
    report_sections = []

    for agent_key, report_text in agent_reports.items():
        agent = AGENT_DEFINITIONS[agent_key]
        # The coordinator does not need every token from every specialist.
        # A capped excerpt keeps the local model responsive and avoids flooding context.
        report_excerpt = str(report_text).strip()[:5500]
        report_sections.append(
            f"### {agent['name']}\n{report_excerpt}"
        )

    reports_text = "\n\n".join(report_sections)

    return f"""
You are the Coordinator Agent for a local/offline analytics team.

Your job is to combine the specialist reports into one practical action plan.
You are training the user to think like a SQL analyst, Power BI developer,
BI analyst, operations analyst, and data quality analyst.

Keep SQL Server access read-only. Do not suggest INSERT, UPDATE, DELETE, DROP,
ALTER, CREATE, EXEC, or destructive commands.

User question:
{user_question}

Current selection:
{evidence["selection_description"]}

Database:
{evidence["selected_database"]}

Table:
{evidence["selected_table"]}

Loaded rows:
{evidence["row_count"]}

Column count:
{evidence["column_count"]}

Selected targets:
{evidence["selected_targets_text"]}

Columns:
{evidence["columns_text"]}

Operational BI summary, if available:
{evidence["risk_summary_text"]}

Power BI output files, if available:
{evidence["powerbi_files_text"]}

Agent SQL execution rules:
{evidence["sql_rules_text"]}

SQL Server context memory:
{evidence.get("sql_server_memory_text", "SQL Server context memory was not provided.")}

Retrieved RAG context:
{rag_context}

Specialist agent reports:
{reports_text}

Required answer format:

## Executive Summary
Explain the main finding in plain business language.

## Analytics Frame
State the business question, data grain, core metrics, key segments, and assumptions.

## Agent Consensus
Summarize what the specialists agree on.

## Disagreements Or Uncertainties
List assumptions that need verification.

## Priority Action Plan
Give a numbered plan for the next practical work.

## SQL Workbench
Provide the best read-only SQL Server query drafts from the team, if any.

## Power BI / Analytics Plan
Recommend pages, visuals, DAX ideas, and exported files.

## Analytics Training Notes
Explain what data analytics, BI, and operational decision-making skill this teaches.

## Executive Story
Describe how this should be presented to a manager or interview audience.

## Validation Checklist
List what must be checked before trusting the analysis.
""".strip()


def ensure_agent_reports_dir():
    os.makedirs(AGENT_REPORTS_DIR, exist_ok=True)
    os.makedirs(AGENT_HISTORY_DIR, exist_ok=True)
    return AGENT_REPORTS_DIR


def safe_filename_part(value):
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(value)
    )
    return cleaned.strip("_") or "agent"


def save_agent_report(agent_key, report_text, metadata, sql_text=None):
    output_dir = ensure_agent_reports_dir()
    agent = AGENT_DEFINITIONS[agent_key]
    timestamp = datetime.now().isoformat(timespec="seconds")
    timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(output_dir, agent["report_file"])
    manifest_path = os.path.join(output_dir, "agent_run_manifest.json")
    history_report_path = os.path.join(
        AGENT_HISTORY_DIR,
        f"{timestamp_for_file}_{safe_filename_part(agent_key)}.md",
    )
    history_manifest_path = os.path.join(
        AGENT_HISTORY_DIR,
        f"{timestamp_for_file}_{safe_filename_part(agent_key)}_manifest.json",
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    with open(history_report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    sql_path = None
    history_sql_path = None
    if sql_text:
        sql_path = os.path.join(output_dir, "sql_agent_queries.sql")
        with open(sql_path, "w", encoding="utf-8") as file:
            file.write(sql_text.strip() + "\n")

        history_sql_path = os.path.join(
            AGENT_HISTORY_DIR,
            f"{timestamp_for_file}_{safe_filename_part(agent_key)}.sql",
        )
        with open(history_sql_path, "w", encoding="utf-8") as file:
            file.write(sql_text.strip() + "\n")

    manifest = {
        "generated_at": timestamp,
        "agent_key": agent_key,
        "agent_name": agent["name"],
        "report_path": report_path,
        "history_report_path": history_report_path,
        "sql_path": sql_path,
        "history_sql_path": history_sql_path,
        "metadata": metadata,
        "notes": [
            "Generated locally by the Streamlit AI Agent Workspace.",
            "No SQL Server data was modified.",
            "Review generated SQL before running it.",
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    with open(history_manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "report": report_path,
        "report_history": history_report_path,
        "sql": sql_path,
        "sql_history": history_sql_path,
        "manifest": manifest_path,
        "manifest_history": history_manifest_path,
    }


def save_agent_team_report(team_report_text, individual_reports, metadata, sql_text=None):
    output_dir = ensure_agent_reports_dir()
    timestamp = datetime.now().isoformat(timespec="seconds")
    timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(output_dir, "agent_team_report.md")
    history_report_path = os.path.join(
        AGENT_HISTORY_DIR,
        f"{timestamp_for_file}_agent_team_report.md",
    )
    summary_path = os.path.join(output_dir, "agent_team_summary.json")
    history_summary_path = os.path.join(
        AGENT_HISTORY_DIR,
        f"{timestamp_for_file}_agent_team_summary.json",
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(team_report_text)

    with open(history_report_path, "w", encoding="utf-8") as file:
        file.write(team_report_text)

    sql_path = None
    history_sql_path = None
    if sql_text:
        sql_path = os.path.join(output_dir, "agent_team_sql_drafts.sql")
        history_sql_path = os.path.join(
            AGENT_HISTORY_DIR,
            f"{timestamp_for_file}_agent_team_sql_drafts.sql",
        )
        with open(sql_path, "w", encoding="utf-8") as file:
            file.write(sql_text.strip() + "\n")
        with open(history_sql_path, "w", encoding="utf-8") as file:
            file.write(sql_text.strip() + "\n")

    summary = {
        "generated_at": timestamp,
        "team_order": AGENT_TEAM_ORDER,
        "report_path": report_path,
        "history_report_path": history_report_path,
        "sql_path": sql_path,
        "history_sql_path": history_sql_path,
        "metadata": metadata,
        "individual_reports": {
            agent_key: {
                "agent_name": AGENT_DEFINITIONS[agent_key]["name"],
                "characters": len(str(report_text)),
            }
            for agent_key, report_text in individual_reports.items()
        },
        "notes": [
            "Generated locally by the Streamlit AI Agent Team.",
            "Specialist reports were synthesized by the Coordinator Agent.",
            "No SQL Server data was modified.",
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(history_summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return {
        "team_report": report_path,
        "team_report_history": history_report_path,
        "team_sql": sql_path,
        "team_sql_history": history_sql_path,
        "team_summary": summary_path,
        "team_summary_history": history_summary_path,
    }
