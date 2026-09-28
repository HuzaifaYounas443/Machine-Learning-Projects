from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, IntegerField, SelectField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange, Length


class NewScanForm(FlaskForm):
    patient_code = StringField(
        "Existing Patient ID (leave blank to register a new patient)",
        validators=[Optional(), Length(max=20)],
    )
    patient_name = StringField("Patient Name", validators=[DataRequired(), Length(max=120)])
    age = IntegerField("Age", validators=[Optional(), NumberRange(min=0, max=120)])
    gender = SelectField(
        "Gender",
        choices=[("", "Select"), ("Male", "Male"), ("Female", "Female"), ("Other", "Other")],
        validators=[Optional()],
    )
    notes = TextAreaField("Clinical Notes (optional)", validators=[Optional(), Length(max=2000)])
    image = FileField(
        "X-Ray Image",
        validators=[
            FileRequired(message="Please select an X-ray image."),
            FileAllowed(["jpg", "jpeg", "png"], "Only JPG and PNG images are supported."),
        ],
    )
    submit = SubmitField("Run Analysis")
