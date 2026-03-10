from __future__ import annotations

from flask import Flask, send_from_directory

from inkwell.db import DEFAULT_DB_PATH
from inkwell.web.routes.ingest import ingest_bp
from inkwell.web.routes.db_viewer import db_viewer_bp
from inkwell.web.routes.annotate import annotate_bp


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path
    
    app.register_blueprint(ingest_bp)
    app.register_blueprint(db_viewer_bp)
    app.register_blueprint(annotate_bp)
    
    @app.route("/")
    def index():
        return '<h1>Inkwell</h1><ul><li><a href="/ingest">Ingestion Review</a></li><li><a href="/annotate">Line Annotation</a></li><li><a href="/db">Database Viewer</a></li></ul>'
    
    @app.route("/working/<path:filename>")
    def serve_working(filename):
        """Serve files from the working directory."""
        # Get working directory from DEFAULT_DB_PATH
        working_dir = DEFAULT_DB_PATH.parent
        return send_from_directory(working_dir, filename)
    
    @app.teardown_appcontext
    def close_connection(exception):
        pass
    
    return app
