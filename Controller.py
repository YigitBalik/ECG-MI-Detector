from flask import render_template, request, redirect, url_for, flash
from pipeline.pdf2data import DataExtractor
from pipeline.model import Model
from pipeline import explain
from forms import *
from werkzeug.utils import secure_filename
import numpy as np
from time import time
import os
from __main__ import app

class Controller(object):

    def __init__(self):
        self.ALLOWED_EXTENSIONS = {'pdf'}
        self.extractor = DataExtractor(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER'])
        self.model = Model()

    def main(self):
        fileUploader = UploadFile()
        return render_template("main.html", fileUploader=fileUploader)
    
    def about(self):
        return render_template("about.html")
    
    def __allowed_file(self, filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def __delete_uploads(self, filename):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4], "signals.npy"))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4], "info.npy"))
        os.rmdir(os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4]))
    
    def execute(self):
        fileUploader = UploadFile()
        if fileUploader.validate_on_submit():
            file = request.files['file']
            if file.name == '':
                flash('No selected file')
                return redirect(request.url)
            if file and self.__allowed_file(file.filename):
                start_time = time()
                filename = secure_filename(fileUploader.file.data.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                self.extractor.extract(filename)

                signals = np.load(os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4], "signals.npy"), allow_pickle=True)
                info = np.load(os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4], "info.npy"), allow_pickle=True)
                cleaned_exg, info, pred = self.model.predict(signals, info)
                diag = "MI" if np.argmax(pred, axis=1) else "Not MI"
                explain.interpret(filename[:-4], cleaned_exg, info, pred)
                duration = time() - start_time
                if info[0,0] == 0:
                    patient_info = ["-", "-"]
                else:
                    patient_info = np.int8(np.squeeze(info))

                self.__delete_uploads(filename)
                return render_template("results.html", 
                                       patient_info = patient_info, 
                                       diag = diag, 
                                       duration = round(duration,2))

        return render_template("main.html", fileUploader=fileUploader)