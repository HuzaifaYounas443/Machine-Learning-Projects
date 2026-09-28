import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-in-production")

    # SQLite by default. Set DATABASE_URL to a postgresql:// URI to switch to Postgres.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "app", "static", "results")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload cap

    # Path to the two trained model checkpoints. Drop your .pth files in models/
    # or override these with environment variables.
    STAGE1_MODEL_PATH = os.environ.get(
        "STAGE1_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "best_model_Stage1_Tumor_convnext.pth"),
    )
    STAGE2_MODEL_PATH = os.environ.get(
        "STAGE2_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "final_model_Stage2_Benign_vs_Malignant_convnext.pth"),
    )

    # Same target layer used for Grad-CAM during training/evaluation
    GRADCAM_TARGET_LAYER = "backbone.features.7.2.block.6"
