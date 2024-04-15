import os
import cv2 as cv
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras import models
import utils

UPLOAD_FOLDER = './upload_folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'justfortestingsowhatever'

model = models.load_model("checkpoints/model.keras")

@app.route("/", methods=["GET", "POST"])
def upload_image():
  if request.method == 'POST':
    if 'file' not in request.files:
      flash('No file part')
      return "None"
    file = request.files['file']
    
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    im = np.expand_dims(im, axis=0)
    return utils.dict_letras[str(np.argmax(model.predict(im)) + 1).zfill(3)]
  return "a"