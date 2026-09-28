import os

from flask import Flask

from .extensions import db, login_manager, csrf


def create_app(config_object=None):
    app = Flask(__name__)

    if config_object is None:
        from config import Config as config_object
    app.config.from_object(config_object)

    # Ensure required folders exist
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.makedirs(os.path.join(base_dir, "instance"), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    from .auth.routes import auth_bp
    from .main.routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()

    @app.context_processor
    def inject_now():
        from datetime import datetime
        return {"current_year": datetime.utcnow().year}

    return app
