from __future__ import annotations

from flask import Flask

from inkwell.web.routes.ingest import ingest_bp
from inkwell.web.routes.db_viewer import db_viewer_bp


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path
    
    app.register_blueprint(ingest_bp)
    app.register_blueprint(db_viewer_bp)
    
    @app.route("/")
    def index():
        return '<h1>Inkwell</h1><ul><li><a href="/ingest">Ingestion Review</a></li><li><a href="/db">Database Viewer</a></li></ul>'
    
    @app.teardown_appcontext
    def close_connection(exception):
        pass
    
    return app
