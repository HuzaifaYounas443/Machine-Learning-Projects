from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from .extensions import db, login_manager


class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    scans = db.relationship("Scan", backref="clinician", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Patient(db.Model):
    __tablename__ = "patient"

    id = db.Column(db.Integer, primary_key=True)
    patient_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    scans = db.relationship(
        "Scan", backref="patient", lazy=True, order_by="Scan.created_at.desc()"
    )

    @property
    def scan_count(self):
        return len(self.scans)


class Scan(db.Model):
    __tablename__ = "scan"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    original_image = db.Column(db.String(255))

    # Stage 1: tumor detection
    tumor_detected = db.Column(db.Boolean, default=False)
    tumor_confidence = db.Column(db.Float)
    tumor_gradcam = db.Column(db.String(255))

    # Stage 2: benign vs malignant (only populated if tumor_detected is True)
    classification = db.Column(db.String(20))  # 'Benign' or 'Malignant'
    classification_confidence = db.Column(db.Float)
    classification_gradcam = db.Column(db.String(255))

    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def status_label(self):
        if not self.tumor_detected:
            return "No Tumor Detected"
        if self.classification:
            return self.classification
        return "Tumor Detected"
