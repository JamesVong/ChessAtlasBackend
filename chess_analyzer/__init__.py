import os

from flask import Flask, jsonify
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app)

    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "8"))
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return jsonify({
            "status": "error",
            "message": f"Image is too large. Max allowed size is {max_upload_mb} MB.",
        }), 413

    with app.app_context():
        from .api import routes

    print("Flask app created and models loaded.")
    return app
