from flask import Flask
import os
app = Flask(__name__)
app.secret_key = "in development"
app.config["UPLOAD_FOLDER"] = "./Uploads"
app.config["MODEL_PATH"] = "./models"
app.config['WTF_CSRF_TIME_LIMIT'] = 86400
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.mkdir(app.config["UPLOAD_FOLDER"])

from Controller import Controller
controller = Controller()



app.add_url_rule("/", endpoint="main", view_func=controller.main, methods=['GET'])
app.add_url_rule("/about", endpoint="about", view_func=controller.about, methods=['GET'])
app.add_url_rule('/results/', endpoint="results", view_func=controller.execute, methods=['POST'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))