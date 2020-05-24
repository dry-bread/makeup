import os
from flask import Flask, flash, request, redirect, url_for,escape,send_from_directory, session
from werkzeug.utils import secure_filename
import uuid
from User_Face_Chek import Face_Chek
from flask import jsonify

# $env:FLASK_APP = "Makeup.py"
# python -m flask run
UPLOAD_FOLDER = 'photo'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'zhinengmeizhaung'

def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS, filename.rsplit('.', 1)[1].lower()

@app.route('/',methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        return 'hello'
    else:
        return 'word'

@app.route('/userface',methods=['GET', 'POST'])
def user_face():
    print('userface recived')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        allowed, suffix = allowed_file(file.filename)
        if file and allowed:
            filename = str(uuid.uuid1()) + '.' + suffix
            session['filename'] = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            checkResult=Face_Chek.Fcae_detect(UPLOAD_FOLDER+"/"+filename)
            if checkResult ==(0,0,0,0):
                return "change photo"
            return "nice photo"
    return "error"

@app.route('/userface/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/targetface',methods=['GET'])
def target_face():
    print('targetface recived')
    styleName = request.args['style']
    print(styleName)
    return jsonify(("6828b8d2-5c89-11ea-a43a-3cf862ea7a5f.jpg", "ed614508-601b-11ea-8c30-3cf862ea7a5f.jpg"))


with app.test_request_context():
    print(url_for('hello_world',_external=True))
    print(url_for('user_face'))
    print(url_for('target_face'))
    