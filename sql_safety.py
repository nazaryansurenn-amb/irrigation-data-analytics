"""Read-only check for SQL written by the local model before it reaches SQL Server.

T-SQL does not need a semicolon between statements: "SELECT 1 SHUTDOWN" is a
SELECT followed by a SHUTDOWN. So it is not enough that a query starts with
SELECT. The check below blocks every keyword that can start another statement.
They are all T-SQL reserved words, so none can be an unquoted column name.

Comments, string literals and quoted identifiers are removed first, the way SQL
Server reads them, so text inside them is neither blocked ("WHERE Status =
'in use'") nor able to hide a statement.

This is a filter, not a security boundary. For real protection, run the app
under a SQL Server login that can only read (db_datareader).
"""

import re

# Keywords that start a statement other than SELECT / WITH, plus INTO
# (SELECT ... INTO creates a table) and the rowset functions that reach
# outside the database. FETCH is not listed: OFFSET ... FETCH is valid SELECT.
BLOCKED_KEYWORDS = frozenset(
    """
    alter backup begin break bulk checkpoint close commit continue create
    dbcc deallocate declare delete deny disable drop dump enable exec execute
    goto grant if insert into kill load merge open opendatasource openquery
    openrowset print raiserror readtext reconfigure restore return revert
    revoke rollback save set setuser shutdown truncate update updatetext use
    waitfor while writetext
    """.split()
)

# Extended and system stored procedures (xp_cmdshell, sp_executesql, ...).
BLOCKED_PREFIXES = ("xp_", "sp_")

# @variables and #temp tables are one token each, so "@set" is not SET.
_WORD = re.compile(r"[a-z_@#][a-z0-9_@#$]*")


class _UnterminatedError(ValueError):
    pass


def _code_only(sql_query):
    """Return the query with comments removed and the contents of string
    literals and quoted identifiers blanked out, following T-SQL lexing:
    block comments nest, '' escapes a quote in a string, ]] escapes ] in a
    bracketed identifier."""
    out = []
    i, n = 0, len(sql_query)
    while i < n:
        if sql_query.startswith("--", i):
            end = sql_query.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif sql_query.startswith("/*", i):
            depth, i = 1, i + 2
            while depth:
                if i >= n:
                    raise _UnterminatedError("comment")
                if sql_query.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql_query.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif sql_query[i] in "'\"[":
            close = "]" if sql_query[i] == "[" else sql_query[i]
            j = i + 1
            while True:
                k = sql_query.find(close, j)
                if k == -1:
                    raise _UnterminatedError("string or identifier")
                if sql_query.startswith(close * 2, k):
                    j = k + 2
                    continue
                break
            out.append(" '' " if close == "'" else " x ")
            i = k + 1
        else:
            out.append(sql_query[i])
            i += 1
    return "".join(out)


def validate_read_only_sql(sql_query):
    try:
        code = _code_only(sql_query or "")
    except _UnterminatedError as error:
        return False, f"The SQL query has an unterminated {error}."

    normalized_query = re.sub(r"\s+", " ", code).strip().lower()

    if not normalized_query:
        return False, "The SQL query is empty."

    if not (normalized_query.startswith("select ") or normalized_query.startswith("with ")):
        return False, "Only SELECT queries are allowed."

    if ";" in normalized_query.rstrip("; "):
        return False, "Only one SQL statement is allowed."

    for word in _WORD.findall(normalized_query):
        if word in BLOCKED_KEYWORDS or word.startswith(BLOCKED_PREFIXES):
            return False, "This query contains a blocked keyword or command."

    return True, "Query is read-only."
