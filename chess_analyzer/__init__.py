from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    with app.app_context():
        from .api import routes
    
    print("✅ Flask App created and models loaded.")
    return app