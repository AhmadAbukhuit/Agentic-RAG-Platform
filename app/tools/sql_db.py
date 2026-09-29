import sqlite3

from langchain_core.tools import tool


@tool
def execute_sql_query(query: str, db_path: str | None = "analytics.db") -> str:
    """Executes a read-only SQL query against the relational database and returns formatted rows.
    
    Args:
        query: The SELECT SQL query string to execute.
        db_path: Path to the SQLite database file.
    """
    clean_query = query.strip()
    # Guard against accidental destructive queries
    lower_query = clean_query.lower()
    destructive_keywords = ["drop", "delete", "truncate", "update", "insert", "alter"]
    if any(keyword in lower_query.split() for keyword in destructive_keywords):
        return "Error: Only read-only SELECT queries are permitted for agent tools."

    try:
        conn = sqlite3.connect(db_path or "analytics.db")
        cursor = conn.cursor()
        cursor.execute(clean_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "Query executed successfully. 0 rows returned."

        # Format tabular output
        formatted_result = f"Columns: {', '.join(columns)}\n"
        for row in rows[:50]:  # Limit output size to prevent context overflow
            formatted_result += f"{row}\n"
        return formatted_result
    except Exception as e:
        return f"Database error executing SQL query: {e!s}"
