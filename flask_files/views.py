from flask import render_template, request
import pandas as pd
from pandas import DataFrame
from werkzeug import secure_filename
from flask_files import app

# specify upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
# specify allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'csv', 'xls', 'xlsx'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
    theFile = request.files['file']
    if theFile and allowed_file(theFile.filename):
        theFile.save(secure_filename(theFile.filename))
        from flask_files import featureFunctions
        intRep, tb1, opNF, claComp, imgs = featureFunctions.preProcess(theFile.filename)
        return render_template('uploader.html',intRep = intRep, tb1 = tb1, opNF = opNF, claComp = claComp, imgs = imgs)
    elif theFile and not allowed_file(theFile.filename):
        return 'Wrong File Format'
    elif not theFile:
        return 'No File Uploaded'
