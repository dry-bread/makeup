import os
from flask import Flask, flash, request, redirect, url_for,escape,send_from_directory, session
from werkzeug.utils import secure_filename
import uuid
from User_Face_Chek import Face_Chek
from flask import jsonify
import style
from Face_Module import make
import hashlib

# $env:FLASK_APP = "Makeup.py"
# python -m flask run
UPLOAD_FOLDER = 'static/photo'
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            md5 = hashlib.md5()
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as saved_file:
                md5.update(saved_file.read())
            md5_filename = str(md5.hexdigest())+ '.' + suffix
            print(md5_filename)
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], md5_filename)):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return "nice photo:"+md5_filename
            
            os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], md5_filename))
            checkResult=Face_Chek.Fcae_detect(os.path.join(app.config['UPLOAD_FOLDER'], md5_filename))
            if checkResult ==(0,0,0,0):
                return "change photo"
            return "nice photo:"+md5_filename
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
    return jsonify(style.style[styleName])

@app.route('/tutorial', methods=['GET'])
def tutorial():
    faceFile = request.args['faceFile']
    targetFile = request.args['targetFile']
    print(faceFile, targetFile)

    suffix = faceFile.rsplit('.', 1)[1].lower()

    res_filename = faceFile + "-" + targetFile + "zhexia." + suffix
    if(os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], res_filename))):
        return "OK"

    res = make.makeup(faceFile, targetFile)
    print(res)
    return "OK"


with app.test_request_context():
    print(url_for('hello_world',_external=True))
    print(url_for('user_face'))
    print(url_for('target_face'))
    