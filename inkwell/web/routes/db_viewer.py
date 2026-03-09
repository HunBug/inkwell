from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, current_app, g

from inkwell.db import get_connection


db_viewer_bp = Blueprint("db_viewer", __name__, url_prefix="/db")


def get_db():
    """Get database connection for current request."""
    if "db" not in g:
        g.db = get_connection(current_app.config.get("DB_PATH"))
    return g.db


@db_viewer_bp.route("/")
def index():
    """Show database overview."""
    db = get_db()
    
    # Get table names
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    
    return render_template("db_viewer.html", tables=[t["name"] for t in tables])


@db_viewer_bp.route("/query", methods=["GET", "POST"])
def query_executor():
    """Execute custom SQL queries (SELECT only)."""
    db = get_db()
    results = None
    error = None
    query = None
    row_count = 0
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        
        if not query:
            error = "Please enter a query"
        elif not query.upper().startswith("SELECT"):
            error = "Only SELECT queries are allowed"
        else:
            try:
                cursor = db.execute(query)
                results = cursor.fetchall()
                row_count = len(results)
            except Exception as e:
                error = str(e)
    
    return render_template(
        "db_query.html",
        query=query,
        results=results,
        error=error,
        row_count=row_count,
    )


@db_viewer_bp.route("/table/<table_name>")
def view_table(table_name: str):
    """View contents of a specific table."""
    db = get_db()
    
    # Get schema
    schema = db.execute(f"PRAGMA table_info({table_name})").fetchall()
    
    # Get row count
    count_result = db.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()
    total_count = count_result["cnt"]
    
    # Get data (limit to 100 rows)
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    rows = db.execute(f"SELECT * FROM {table_name} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    
    return render_template(
        "db_table.html",
        table_name=table_name,
        schema=schema,
        rows=rows,
        total_count=total_count,
        limit=limit,
        offset=offset,
    )
