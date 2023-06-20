from flask.app import Flask
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms.validators import DataRequired

class UploadFile(FlaskForm):
    file = FileField(validators=[DataRequired()])