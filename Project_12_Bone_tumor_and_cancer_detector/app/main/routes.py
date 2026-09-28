import os
from datetime import datetime

from flask import Blueprint, render_template, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import Patient, Scan
from .forms import NewScanForm
from ..ml.inference import get_models, run_pipeline

main_bp = Blueprint("main", __name__, template_folder="../templates/main")


def _generate_patient_code():
    last = Patient.query.order_by(Patient.id.desc()).first()
    next_id = (last.id + 1) if last else 1
    return f"PT-{next_id:05d}"


@main_bp.route("/")
def index():
    return redirect(url_for("main.dashboard"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    stats = {
        "total_scans": Scan.query.count(),
        "total_patients": Patient.query.count(),
        "tumor_positive": Scan.query.filter_by(tumor_detected=True).count(),
        "malignant_count": Scan.query.filter_by(classification="Malignant").count(),
        "benign_count": Scan.query.filter_by(classification="Benign").count(),
    }
    recent_scans = Scan.query.order_by(Scan.created_at.desc()).limit(8).all()
    return render_template("main/dashboard.html", stats=stats, recent_scans=recent_scans)


@main_bp.route("/scan/new", methods=["GET", "POST"])
@login_required
def new_scan():
    form = NewScanForm()

    if form.validate_on_submit():
        patient = None
        if form.patient_code.data:
            patient = Patient.query.filter_by(patient_code=form.patient_code.data.strip()).first()
            if patient is None:
                flash("No patient found with that ID. A new patient record will be created.", "error")

        if patient is None:
            patient = Patient(
                patient_code=_generate_patient_code(),
                name=form.patient_name.data.strip(),
                age=form.age.data,
                gender=form.gender.data or None,
            )
            db.session.add(patient)
            db.session.flush()
        else:
            patient.name = form.patient_name.data.strip()
            if form.age.data:
                patient.age = form.age.data
            if form.gender.data:
                patient.gender = form.gender.data

        image_file = form.image.data
        filename = secure_filename(image_file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
        image_file.save(upload_path)

        try:
            stage1_model, stage2_model = get_models(current_app.config["STAGE1_MODEL_PATH"])
            result = run_pipeline(
                upload_path,
                stage1_model,
                stage2_model,
                current_app.config["RESULTS_FOLDER"],
                current_app.config["GRADCAM_TARGET_LAYER"],
                current_app.config["STAGE2_MODEL_PATH"],
            )
        except Exception as exc:
            current_app.logger.exception("Inference failed")
            db.session.rollback()
            flash(f"Analysis failed: {exc}", "error")
            return redirect(url_for("main.new_scan"))

        scan = Scan(
            patient_id=patient.id,
            user_id=current_user.id,
            original_image=unique_name,
            tumor_detected=result["tumor_detected"],
            tumor_confidence=result["tumor_confidence"],
            tumor_gradcam=result["tumor_gradcam"],
            classification=result["classification"],
            classification_confidence=result["classification_confidence"],
            classification_gradcam=result["classification_gradcam"],
            notes=form.notes.data,
        )
        db.session.add(scan)
        db.session.commit()

        flash("Analysis complete.", "success")
        return redirect(url_for("main.result", scan_id=scan.id))

    return render_template("main/new_scan.html", form=form)


@main_bp.route("/scan/<int:scan_id>")
@login_required
def result(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    return render_template("main/result.html", scan=scan)


@main_bp.route("/history")
@login_required
def history():
    scans = Scan.query.order_by(Scan.created_at.desc()).all()
    return render_template("main/history.html", scans=scans)


@main_bp.route("/patient/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template("main/patient_detail.html", patient=patient)
