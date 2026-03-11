from __future__ import annotations

import os

from flask import Flask, render_template, send_from_directory

from inkwell.db import DEFAULT_DB_PATH
from inkwell.web.routes.ingest import ingest_bp
from inkwell.web.routes.db_viewer import db_viewer_bp
from inkwell.web.routes.annotate import annotate_bp
from inkwell.web.routes.jobs import jobs_bp


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path
    app.config["INKWELL_SHARED"] = os.environ.get("INKWELL_SHARED")

    app.register_blueprint(ingest_bp)
    app.register_blueprint(db_viewer_bp)
    app.register_blueprint(annotate_bp)
    app.register_blueprint(jobs_bp)
    
    @app.route("/")
    def index():
        return render_template("home.html")
    
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
